import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

from scripts.torch_utils import use_gpu_if_possible
from scripts.train_utils import AverageMeter


## Dataset

class InvalidDatasetException(Exception):
    
    def __init__(self,len_of_paths,len_of_labels):
        super().__init__( f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})" )


transform = transforms.Compose([transforms.ToTensor()])


class AnimalDataset(Dataset):
    
    def __init__(self, img_paths, img_labels, size_of_images):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.size_of_images = size_of_images
        if len(self.img_paths) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths,self.img_labels)
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        PIL_IMAGE = Image.open(self.img_paths[index]).resize(self.size_of_images)
        tensor_image = transform(PIL_IMAGE)
        label = self.img_labels[index]
        
        return tensor_image, label


paths = []
labels = []
label_map = {2:"Wild"}
for path in glob('/u/dssc/mmarturi/Deep_Learning/AnimalsBetaVAE/animalsBetaVAE/afhq/train/wild/*') + glob('/u/dssc/mmarturi/Deep_Learning/AnimalsBetaVAE/animalsBetaVAE/afhq/val/wild/*'):
    paths.append(path)
    labels.append(2)

dataset = AnimalDataset(paths, labels, (250,250))
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)


########### Model ################

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAEEncoder(nn.Module):
    def __init__(self, dim_latent:int, nc:int): # nc is 3 
        super().__init__()
        self.encoder = nn.Sequential( # input: size_batch(B), nc=3, 250, 250
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(nc, 32, 4, 2, 1),      #B, 32, 125, 125 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 2, 2),      #B, 32, 64, 64 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, 2, 1),      #B, 32, 32, 32 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, 2, 1),      #B, 32, 16, 16 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1),      #B, 64, 8, 8 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1),      #B, 64, 4, 4 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 4, 1),        #B, 256, 1, 1 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            View((-1, 256*1*1)),
        )
        self.encode_mu = nn.Linear(256, dim_latent)
        self.encode_sigma = nn.Linear(256, dim_latent)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        h = self.encoder(X)
        latent_mu = self.encode_mu(h)
        latent_log_sigma = self.encode_sigma(h)
        return latent_mu, latent_log_sigma

class VAEDecoder(nn.Module):
    def __init__(self, dim_latent:int, nc:int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 256),       #256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            View((-1,256,1,1)),                   #B, 256, 1, 1
            nn.ConvTranspose2d(256, 64, 4),       #B, 64, 4, 4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  #B, 64, 8, 8 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  #B, 32, 16, 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  #B, 32, 32, 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  #B, 32, 64, 64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 2, 2),  #B, 32, 125, 125 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  #B, 3, 250, 250
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    
    def forward(self, z:torch.Tensor) -> tuple:
        reconstruction = self.decoder(z)
        return reconstruction

class VAE(nn.Module):
    def __init__(self, dim_latent:int, nc:int):
        super().__init__()
        self.encoder = VAEEncoder(dim_latent, nc)
        self.decoder = VAEDecoder(dim_latent, nc)
    
    def sample_latent(self, mu, log_sigma):
        # with reparametrization trick
        device = next(self.parameters()).device
        # white_noise is the epsilon (~N(0,1))
        white_noise = torch.randn(mu.shape).to(device)
        return mu + (log_sigma / 2).exp() * white_noise

    def forward(self, X):
        mu, log_sigma = self.encoder(X)
        z = self.sample_latent(mu, log_sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_sigma


def reconstruction_loss(output, ground_truth):
    reconstruction_loss_fn = nn.BCELoss(reduction="sum")
    return reconstruction_loss_fn(output, ground_truth)

def kl_vae(mu, log_sigma):
    # calculated in close form
    kl = .5 * (log_sigma.exp() ** 2 + mu ** 2 - 1) - log_sigma
    return kl.sum()

def vae_loss(output, mu, log_sigma, ground_truth):
    rec_loss_val = reconstruction_loss(output, ground_truth)
    beta = 120
    
    kl_loss_val = kl_vae(mu, log_sigma)
    return rec_loss_val + beta*kl_loss_val


######### Training ############

latent_dim = 32
nc = 3
vae = VAE(latent_dim, nc)
num_epochs = 50
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
device = use_gpu_if_possible()
ite_print = len(train_loader)


vae.train()
vae = vae.to(device)
flatten = nn.Flatten()
start = time.time()
for epoch in range(num_epochs):
    loss_meter = AverageMeter()
    for i, (X, _) in enumerate(train_loader):
        X = X.to(device)

        optimizer.zero_grad()

        X_hat, mu, log_sigma = vae(X)
        loss = vae_loss(flatten(X_hat), mu, log_sigma, flatten(X))
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), X.shape[0])

        if (i + 1) % ite_print == 0 or (i + 1) == len(train_loader):
            print(f"Epoch {epoch+1} | Loss {loss_meter.avg}")

end = time.time()
print("Training is done!\n")
print(f"Elapsed time for training:{(end-start)/60} minutes.\n")

torch.save(vae.state_dict(), "wildVAEb120.pt")
