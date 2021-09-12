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

#alpha = 1
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
        tran = transforms.Compose([transforms.ToTensor()])
        d1 = tran(mpimg.imread('test2_pic_v2/2898.png')).view(-1, 1024)
        Z_a = self.encode(d1)
        z_a_centoid, logvara = self.gaussian_param_projection(Z_a)
        z_a = self.reparameterize(z_a_centoid, logvara)
        d2 = tran(mpimg.imread('test2_pic_v2/2904.png')).view(-1, 1024)
        Z_b = self.encode(d2)
        z_b_centoid, logvarb = self.gaussian_param_projection(Z_b)
        z_b = self.reparameterize(z_b_centoid, logvarb)
        z_b2a = z_b - z_a
        # Manipulate x_c
        z_c_interp = z_a + alpha * z_b2a
        return z_c_interp, 0.5*(z_a_centoid+z_b_centoid), 0.5*(logvara+logvarb)

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

#test_loader = torch.utils.data.DataLoader(Data(transform=transforms.Compose([transforms.ToTensor()]),train = False),batch_size=batch_size, shuffle=True)

model2 = torch.load('100vae.pkl')
device='cpu'
tran = transforms.Compose([transforms.ToTensor()])
d1 = tran(mpimg.imread('test2_pic_v2/2898.png'))
alpha = 0.0
f = plt.figure()
for i in range(10):
    recon_batch, mu, logvar = model2(d1.to(device).view(-1, 1024))
    temp = recon_batch.view(recon_batch.size(0) // 3, 3, 32, 32).permute(0, 2, 3, 1).detach().numpy()
    img = temp
    in_data = np.asarray(img.reshape(32, 32, 3) * 255, dtype=np.uint8)
    mpimg.imsave('100interpolation_v1/inter_' + str(i+1) + '.png', in_data)
    f.add_subplot(1, 10, i+1)
    plt.imshow(mpimg.imread('100interpolation_v1/inter_' + str(i+1) + '.png'))
    plt.axis('off')
    alpha = alpha + 0.1

plt.show(block=True)
plt.savefig('100interpolation_v1/inter.png')

