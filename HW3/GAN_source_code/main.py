from __future__ import print_function
# %matplotlib inline

# import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from Model import *
# from Customdata import *
import argparse
import os
import matplotlib.image as mpimg
import glob
from Visualization import *
from PIL import Image


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device = torch.device("cpu")


def common_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='./img_align_celeba/', type=str)  # 'data/celeba'
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)#5
    parser.add_argument('--lr', default=0.0002, type=float)

    # print(parser.parse_args([]))
    return parser.parse_args([])


def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
    # Each epoch, we have to go through every data in dataset
    G_losses = []
    D_losses = []
    img_list = []
    iters = 0
    nz = 100
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):
            # initialize gradient for network
            # send the data into device for computation
            generator.zero_grad()
            discriminator.zero_grad()
            real_label = 1
            fake_label = 0
            #print(data.shape)
            real_cpu = data.to(device)
            #real_cpu = data[0].to(device)
            #print(data[0].shape)

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            b_size = real_cpu.size(0)
            #print(real_cpu.shape)
            label = torch.full((b_size,), real_label, device=device)
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            #nz = 100
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_d.step()

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake).view(
                -1)  # Since we just updated D, perform another forward pass of all-fake batch through D
            errG = criterion(output, label)
            # Update your network
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            # Record your loss every iteration for visualization

            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                epoch, num_epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    #fake = generator(noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                torch.save(discriminator.state_dict(), "Dis_" + str(epoch) + "_" + str(i)+".pth")
                torch.save(generator.state_dict(), "Gen_" + str(epoch) + "_" + str(i)+".pth")

            iters += 1

            vis(G_losses, D_losses, img_list,dataloader,device,epoch,i)
            # Remember to save all things you need after all batches finished!!!



class Customdata():
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.path = []
        # self.loader = loader
        for i in range(1,35):
            # self.path.extend(os.path.basename(x) for x in glob.glob(os.path.join(self.root,str(i)+"/*.jpg")))
            # self.path.extend(x for x in glob.glob(os.path.join(self.root,str(i)+"/*.jpg")))
            # print([os.path.basename(x) for x in glob.glob(os.path.join(self.root,str(i)+"/*")])
            # print(os.path.join(self.root,str(i)+"/*.jpg"))
            for x in glob.glob(os.path.join(self.root, str(i) + "/*.jpg")):
                self.path.append(x.split("/")[-2] + "/" + os.path.basename(x))
                # print(x.split("/")[-2]+"/"+os.path.basename(x))
                # break
        # print(self.path)

    def __getitem__(self, index):
        fn = self.path[index]
        #print(os.path.join(self.root, fn))
        # img = self.loader(os.path.join(self.root, fn))
        img = mpimg.imread(os.path.join(self.root, fn))
        # img = mpimg.imread(fn)
        PIL_image = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(PIL_image)
        return img

    def __len__(self):
        #print(self.path)
        #print(len(self.path))
        return len(self.path)


def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    dataset = Customdata(root=args.dataroot, transform=transforms.Compose(
        [transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create the generator and the discriminator()
    # Initialize them
    # Send them to your device
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss(reduction="mean")

    # Start training~~

    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, args.num_epochs)


if __name__ == '__main__':
    args = common_arg_parser()
    # print(args)
    main(args)
