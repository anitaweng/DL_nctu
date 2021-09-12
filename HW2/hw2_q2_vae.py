import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision

#from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import norm
from pathlib import Path
import glob
import imageio
import os
import matplotlib.image as mpimg
#import scipy
from PIL import Image


batch_size = 150
epochs = 1000
device='cpu'

class Data():
    def __init__(self, transform=None, train = True):
        self.path = []
        self.target = []
        if train == True:
            self.root = 'preprocess_train/'
        else:
            self.root = 'preprocess_test/'

        for filename in glob.glob(self.root+'*.png'):
            self.path.append(filename)

        self.transform = transform
        #self.loader = loader

    def __getitem__(self, index):
        fn = self.path[index]
        #img = self.loader(os.path.join(self.root, fn))
        img = os.path.join(fn)
        #print(img)
        if self.transform is not None:
            img = self.transform(mpimg.imread(img))
            #print(img)
        return img

    def __len__(self):
        return len(self.path)


train_loader = torch.utils.data.DataLoader(
    Data(transform=transforms.Compose([transforms.ToTensor()]),train = True),batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(Data(transform=transforms.Compose([transforms.ToTensor()]),train = False),batch_size=batch_size, shuffle=True)

len(train_loader.dataset), len(test_loader.dataset)

class VAEEncoder(nn.Module):
    def __init__(self, data_size, hidden_sizes, latent_size):
        super(VAEEncoder,self).__init__()

        self.data_size=data_size

        # construct the encoder
        encoder_szs = [data_size] + hidden_sizes
        encoder_layers = []

        for in_sz,out_sz, in zip(encoder_szs[:-1], encoder_szs[1:]):
            encoder_layers.append(nn.Linear(in_sz, out_sz))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(encoder_szs[-1], latent_size)
        self.encoder_logvar = nn.Linear(encoder_szs[-1], latent_size)

    def encode(self, x):
        return self.encoder(x)

    def gaussian_param_projection(self, x):
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encode(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, data_size, hidden_sizes, latent_size):
        super(VAEDecoder,self).__init__()

        # construct the decoder
        hidden_sizes = [latent_size] + hidden_sizes
        decoder_layers = []
        for in_sz,out_sz, in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            decoder_layers.append(nn.Linear(in_sz, out_sz))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_sizes[-1], data_size))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, data_size, encoder_szs, latent_size, decoder_szs=None):
        super(VAE,self).__init__()

        # if decoder_szs not specified, assume symmetry
        if decoder_szs is None:
            decoder_szs = encoder_szs[::-1]

        # construct the encoder
        self.encoder = VAEEncoder(data_size=data_size, hidden_sizes=encoder_szs,
                                  latent_size=latent_size)

        # construct the decoder
        self.decoder = VAEDecoder(data_size=data_size, latent_size=latent_size,
                                           hidden_sizes=decoder_szs)

        self.data_size = data_size
        #print(self.data_size)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        p_x = self.decoder(z)
        return p_x, mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    CE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return CE + KLD

def train(train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data = data.view(-1, 1024)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        #print(recon_batch.size())
        loss = loss_function(recon_batch, data.squeeze(), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)

def test(test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        i =0
        for data in test_loader:
            data = data.to(device)
            origin_data = data.view(-1, 1024)
            recon_batch, mu, logvar = model(origin_data)
            test_loss += loss_function(recon_batch, origin_data, mu, logvar).item()
            temp = recon_batch.view(recon_batch.size(0)//3,3,32,32).permute(0, 2, 3, 1).numpy()
            i = i+1
            temp_data = data.view(recon_batch.size(0) // 3, 3, 32, 32).permute(0, 2, 3, 1).numpy()
            for idx in range(0,recon_batch.size(0)//3,50):
                img = temp[idx,:,:,:]
                ori = temp_data[idx,:,:,:]
                in_data = np.asarray(img.reshape(32, 32, 3)*255, dtype=np.uint8)
                in_data_ori = np.asarray(ori.reshape(32, 32, 3) * 255, dtype=np.uint8)
                mpimg.imsave('1000epoch_3/test_' + str(idx) + '_' + str(i) + '.png',in_data)
                mpimg.imsave('1000epoch_3/ori_' + str(idx) +'_' + str(i) + '.png', in_data_ori)
            tran = transforms.Compose([transforms.ToTensor()])
            d = tran(mpimg.imread('preprocess_test/1000.png'))
            recon_batch1, mu1, logvar1 = model(d.to(device).view(-1,1024))
            temp1 = recon_batch1.view(recon_batch1.size(0) // 3, 3, 32, 32).permute(0, 2, 3, 1).numpy()
            img1 = temp1
            in_data1 = np.asarray(img1.reshape(32, 32, 3) * 255, dtype=np.uint8)
            mpimg.imsave('1000epoch_3/test_1000'+ '_' + str(i) +'.png', in_data1)




    return test_loss / len(test_loader.dataset)

def fit(model, epochs):
    train_loss = []
    test_loss = []
    for epoch in range(1, epochs + 1):
        trn_loss = train(train_loader)
        tst_loss = test(test_loader)
        print('epoch {}, train loss: {}, test loss: {}'.format(epoch,round(trn_loss, 6),round(tst_loss, 6)))
        train_loss.append(round(trn_loss, 6))
        test_loss.append(round(tst_loss, 6))

    plt.figure()
    plt.plot(train_loss)
    plt.savefig("1000epoch_3/loss_train.png")
    plt.figure()
    plt.plot(test_loss)
    plt.savefig("1000epoch_3/loss_test.png")

model = VAE(data_size=1024, encoder_szs=[400,150], latent_size=20,
                     decoder_szs=[150,400]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

fit(model, epochs)
torch.save(model, "vae_v2.pkl")