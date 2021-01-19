import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import torchvision
from torch import optim
from torch import nn
import torch.nn.functional as F
import time
import math
import os 
import pickle

g_l_file='gen_loss.sav'
d_l_file='dis_loss.sav'
sample_folder='samples'
si_folder='Space_interpolation'
fi_folder='Factor_interpolation'
saved_models='saved_models'
test_samples='test_samples'
batch_size=64
if(torch.cuda.is_available()):
  device='cuda:0'
else :
  device='cpu'
print('Device for training : {}'.format(device))
torch.pi = torch.acos(torch.zeros(1,device=device)) * 2
train_loader=torch.utils.data.DataLoader(dataset=torchvision.datasets.MNIST('./root',train=True,download=True,transform=torchvision.transforms.ToTensor()) ,batch_size=batch_size,drop_last=True)
test_loader=torch.utils.data.DataLoader(dataset=torchvision.datasets.MNIST('./root',train=False,download=True,transform=torchvision.transforms.ToTensor()) ,batch_size=batch_size,drop_last=True)