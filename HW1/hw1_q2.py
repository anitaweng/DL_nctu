import argparse
import torch #
import torch.nn as nn#
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms#
import tensorflow as tf
import matplotlib.pyplot as plt#
import csv
from torch.autograd import Variable#
import os
import torch.utils.data
import numpy as np
import matplotlib.image as mpimg

loss_train_plot =[]
loss_test_plot =[]
acc_train_plot =[]
acc_test_plot =[]

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 1),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 1), nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(128 * 14 * 14, 500) #5=>128*14*14 3=>128*16*16
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def default_loader(path):
    img_tensor = mpimg.imread(path)
    return img_tensor

class MaskData_train():
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):

        class_names = ['bad good none']
        with open(label, newline='') as csvfile:
            rows = csv.reader(csvfile)
            a = []
            for row in rows:
                a.append(row)
            csvfile.close()
        self.path = []
        self.target = []
        width = np.zeros((3528, 1))
        height = np.zeros((3528, 1))
        xmin = np.zeros((3528, 1))
        ymin = np.zeros((3528, 1))
        xmax = np.zeros((3528, 1))
        ymax = np.zeros((3528, 1))
        for i in range(1, 3529):
            if np.array(a[i])[3] == "bad":
                label_num = 0
            elif np.array(a[i])[3] == "good":
                label_num = 1
            else:
                label_num = 2
            self.target.append(label_num)
            width[i - 1][0] = float(np.array(a[i])[1])
            height[i - 1][0] = float(np.array(a[i])[2])
            xmin[i - 1][0] = float(np.array(a[i])[4])
            ymin[i - 1][0] = float(np.array(a[i])[5])
            xmax[i - 1][0] = float(np.array(a[i])[6])
            ymax[i - 1][0] = float(np.array(a[i])[7])
            self.path.append(str(i-1)+".jpg")


        self.root = root
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.path[index]
        label = self.target[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, (label)

    def __len__(self):
        return len(self.path)

    def getName(self):
        return self.classes

class MaskData_test():
    def __init__(self, root, label, transform=None, target_transform=None,loader=default_loader):
        class_names = ['bad good none']
        with open(label, newline='') as csvfile:
            rows = csv.reader(csvfile)
            a = []
            for row in rows:
                a.append(row)
            csvfile.close()
        self.path = []
        self.target = []
        width = np.zeros((394, 1))
        height = np.zeros((394, 1))
        xmin = np.zeros((394, 1))
        ymin = np.zeros((394, 1))
        xmax = np.zeros((394, 1))
        ymax = np.zeros((394, 1))
        for i in range(1, 395):
            if np.array(a[i])[3] == "bad":
                label_num = 0
            elif np.array(a[i])[3] == "good":
                label_num = 1
            else:
                label_num = 2
            self.target.append(label_num)
            width[i - 1][0] = float(np.array(a[i])[1])
            height[i - 1][0] = float(np.array(a[i])[2])
            xmin[i - 1][0] = float(np.array(a[i])[4])
            ymin[i - 1][0] = float(np.array(a[i])[5])
            xmax[i - 1][0] = float(np.array(a[i])[6])
            ymax[i - 1][0] = float(np.array(a[i])[7])
            self.path.append(str(i - 1) + ".jpg")

        self.root = root
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.path[index]
        label = self.target[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.path)

    def getName(self):
        return self.classes

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    Loss = nn.CrossEntropyLoss()
    correct = 0
    c0=0
    c1=0
    c2=0
    t0 = 0
    t1 = 0
    t2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        g = tf.Graph()
        # add ops to the default graph
        with g.as_default():
            a = pred
            b = target.view_as(pred)
        sess = tf.Session(graph=g)  # session is run on graph g
        sess.run  # run session
        result0 = 0
        result1 = 0
        result2 = 0
        total0 = 0
        total1 = 0
        total2 = 0
        for i in range(0, len(pred)):
            result0 += np.count_nonzero(a[i] == (b[i]) and a[i] == 0)
            result1 += np.count_nonzero(a[i] == (b[i]) and a[i] == 1)
            result2 += np.count_nonzero(a[i] == (b[i]) and a[i] == 2)
            total0 += np.count_nonzero(b[i] == 0)
            total1 += np.count_nonzero(b[i] == 1)
            total2 += np.count_nonzero(b[i] == 2)
        c0+= result0
        c1+= result1
        c2+= result2
        t0+= total0
        t1+= total1
        t2+= total2
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            loss_train_plot.append(loss.item())
    print('\nTrain set:  Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_loader.dataset), 100.*correct / len(train_loader.dataset)))
    print(c0)
    print(c1)
    print(c2)
    print(t0)
    print(t1)
    print(t2)
    print(1.0*c0/t0)
    print(1.0*c1/t1)
    print(1.0*c2/t2)
    acc_train_plot.append(correct / len(train_loader.dataset))
    plt.figure()
    plt.plot(loss_train_plot)
    plt.savefig("loss_train.png")
    plt.figure()
    plt.plot(acc_train_plot)
    plt.savefig("acc_train.png")

def test(args, model, device, test_loader):
    Loss = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    test_loss = 0
    correct = 0
    c0 = 0
    c1 = 0
    c2 = 0
    t0 = 0
    t1 = 0
    t2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += Loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            g = tf.Graph()
            # add ops to the default graph
            with g.as_default():
                a = pred
                b = target.view_as(pred)
            sess = tf.Session(graph=g)  # session is run on graph g
            sess.run  # run session
            result0 = 0
            result1 = 0
            result2 = 0
            total0 = 0
            total1 = 0
            total2 = 0
            for i in range(0, len(pred)):
                result0 += np.count_nonzero(a[i] == (b[i]) and a[i] == 0)
                result1 += np.count_nonzero(a[i] == (b[i]) and a[i] == 1)
                result2 += np.count_nonzero(a[i] == (b[i]) and a[i] == 2)
                total0 += np.count_nonzero(b[i] == 0)
                total1 += np.count_nonzero(b[i] == 1)
                total2 += np.count_nonzero(b[i] == 2)
            c0 += result0
            c1 += result1
            c2 += result2
            t0 += total0
            t1 += total1
            t2 += total2
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc_test_plot.append(100.*(correct/len(test_loader.dataset)))
    print(c0)
    print(c1)
    print(c2)
    print(t0)
    print(t1)
    print(t2)
    print(1.0*c0 / t0)
    print(1.0*c1 / t1)
    print(1.0*c2 / t2)
    loss_test_plot.append(test_loss)
    plt.figure()
    plt.plot(acc_test_plot)
    plt.savefig("acc_test.png")
    plt.figure()
    plt.plot(loss_test_plot)
    plt.savefig("loss_test.png")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=49, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                       help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MaskData_train(root='preprocess_train/',label='train.csv',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,),transforms.Resize((128,128)))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MaskData_test(root='preprocess_test/',label='test.csv',  transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,),transforms.Resize((128,128)))
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = CNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        torch.save(model.state_dict(), "mask_cnn.pt")





if __name__ == '__main__':
    main()
