# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2020/3/2
# @institute: UMSI
# @version: 0.1 alpha

from unet_model import UNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
from data_ldr import *

# Argument
num_epochs = 100
learning_rate = 0.05

# Data Loader
dataset_train = img_seg_ldr()
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)

# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Model
net = UNet(n_channels=3, n_classes=4)

# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), size_average = False)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training
net.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
