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
        #print(z.size())
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
        #p_x = self.decoder(z)
        #p_x = self.decoder(torch.randn(3, 20))
        p_x = self.decoder(torch.randn_like(z))
        #p_x = self.decoder(torch.from_numpy(np.random.normal(size=[3, 20])).float().to(device))
        #p_x = self.decoder(torch.normal(mu, torch.exp(0.5*logvar)))
        #print(torch.randn(mu,logvar).size())
        return p_x, mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    CE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return CE + KLD

device='cpu'
model2 = torch.load('vae.pkl')
tran2 = transforms.Compose([transforms.ToTensor()])
for filename in glob.glob('test_sample_pic/'+'*.png'):
    d2 = tran2(mpimg.imread(filename))
    recon_batch2, mu2, logvar2 = model2(d2.to(device).view(-1,1024))
    temp2 = recon_batch2.view(recon_batch2.size(0)//3, 3, 32, 32).permute(0, 2, 3, 1).detach().numpy()
    img2 = temp2
    in_data2 = np.asarray(img2.reshape(32, 32, 3) * 255, dtype=np.uint8)
    #print('test_sample_result/'+filename.split('/')[-1])
    mpimg.imsave('test_sample_result/'+filename.split('/')[-1], in_data2)


