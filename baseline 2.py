from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms


import glob
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import argparse
parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--dim1',type=int, default = 800,
                    help = 'dimension 1')
parser.add_argument('--dim2',type=int, default = 700,
                    help = 'dimension 2')
parser.add_argument('--epochs',type=int, default = 350,
                    help = 'number of epochs')
epochs = 2000
dim1 = 200
dim2 = 175
l_rate = 0.05


class LinearRegressionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
    # nn.linear is defined in nn.Module
    
    def forward(self, x):
        # Here the forward pass is simply a linear function
        
        out = self.linear(x)
        return out



transform = transforms.Compose(
    [
     transforms.Resize((dim1,dim2)),
     transforms.Grayscale(3),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
     ])

trainset = torchvision.datasets.ImageFolder(root='./data/train', 
                                        transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./data/test', 
                                       transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

train_gray = []
for i in range(len(trainset)):
    train_gray.append(trainset[i])
test_gray = []
for i in range(len(testset)):
    test_gray.append(testset[i])
    
train_grey = []
for i in range(len(train_gray)):
    train_grey.append(train_gray[i][0][0,:,:])
    
train_grey_all = []
for i in range(len(train_gray)):
    train_grey_all.append(train_gray[i][0])
num_train = len(train_grey_all)
    
transform = transforms.Compose(
    [transforms.Resize((dim1,dim2)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
     ])

trainColor = torchvision.datasets.ImageFolder(root='./data/train', 
                                        transform=transform)
trainloader_c = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testColor = torchvision.datasets.ImageFolder(root='./data/test', 
                                       transform=transform)
testloader_c = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

train_color = []
train_R = []
train_G = []
train_B = []
train_target = []
for i in range(len(trainColor)):
    train_color.append(trainColor[i])
test_color = []
for i in range(len(testColor)):
    test_color.append(testColor[i])
for i in range(len(train_color)):
    train_R.append(train_color[i][0][0,:,:])
    train_G.append(train_color[i][0][1,:,:])
    train_B.append(train_color[i][0][2,:,:])
    train_target.append(train_color[i][0])
    

input_dim = dim1*dim2
output_dim = dim1*dim2

grey_tensor = torch.stack(train_grey)
grey_tensor = grey_tensor.view(num_train,dim1*dim2)
#all_tensor = torch.stack(train_target)
#all_tensor = all_tensor.view(num_train,3*dim1*dim2)
r_tensor = torch.stack(train_R)
r_tensor = r_tensor.view(num_train,dim1*dim2)
g_tensor = torch.stack(train_G)
g_tensor = g_tensor.view(num_train,dim1*dim2)
b_tensor = torch.stack(train_B)
b_tensor = b_tensor.view(num_train,dim1*dim2)


model = LinearRegressionModel(input_dim,output_dim)

criterion = nn.MSELoss()# Mean Squared Loss

optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent

#R
for epoch in range(epochs):

    epoch +=1

    inputs = Variable(grey_tensor)
    #labels = Variable(all_tensor)
    labels = Variable(r_tensor)

    #clear grads as discussed in prev post
    optimiser.zero_grad()
    #forward to get predicted values
    outputs = model.forward(inputs)
    loss_r = criterion(outputs, labels)
    loss_r.backward()# back props
    optimiser.step()# update the parameters
    print ('epoch {}, loss {}'.format(epoch,loss_r.data[0]))
    
predicted_R =model.forward(Variable(grey_tensor))


#G
for epoch in range(epochs):
    
    epoch +=1
    
    inputs = Variable(grey_tensor)
    #labels = Variable(all_tensor)
    labels = Variable(g_tensor)
    
    #clear grads as discussed in prev post
    optimiser.zero_grad()
    #forward to get predicted values
    outputs = model.forward(inputs)
    loss_g = criterion(outputs, labels)
    loss_g.backward()# back props
    optimiser.step()# update the parameters
    print ('epoch {}, loss {}'.format(epoch,loss_g.data[0]))

predicted_G =model.forward(Variable(grey_tensor))


#B
for epoch in range(epochs):
    
    epoch +=1
    
    inputs = Variable(grey_tensor)
    #labels = Variable(all_tensor)
    labels = Variable(b_tensor)
    
    #clear grads as discussed in prev post
    optimiser.zero_grad()
    #forward to get predicted values
    outputs = model.forward(inputs)
    loss_b = criterion(outputs, labels)
    loss_b.backward()# back props
    optimiser.step()# update the parameters
    print ('epoch {}, loss {}'.format(epoch,loss_b.data[0]))

predicted_B =model.forward(Variable(grey_tensor))


#put together
overall = torch.FloatTensor(num_train,3,dim1*dim2).zero_()
overall[:,0,:] = predicted_R.data
overall[:,1,:] = predicted_G.data
overall[:,2,:] = predicted_B.data

out = overall.view(num_train,3,dim1,dim2)
torch.save(out,'predicted')




