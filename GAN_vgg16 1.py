#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:04:41 2018

@author: tinghaoli
"""


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

transform = transforms.Compose(
    [
     transforms.Resize((opt.imageSize,opt.imageSize)),
     #transforms.Grayscale(3),
     transforms.ToTensor(),
     #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
     ])

trainset = dset.ImageFolder(root='./data/train',
                                        transform=transform)
assert trainset

#calculate mean and std
train_color = []
train_R = []
train_G = []
train_B = []
for i in range(len(trainset)):
    train_color.append(trainset[i])
for i in range(len(train_color)):
    train_R.append(train_color[i][0][0,:,:])
    train_G.append(train_color[i][0][1,:,:])
    train_B.append(train_color[i][0][2,:,:])
all_train_R = torch.stack(train_R)
mean_R = torch.mean(all_train_R)
std_R = torch.std(all_train_R)

all_train_G = torch.stack(train_G)
mean_G = torch.mean(all_train_G)
std_G = torch.std(all_train_G)

all_train_B = torch.stack(train_B)
mean_B = torch.mean(all_train_B)
std_B = torch.std(all_train_B)


transform = transforms.Compose(
                               [
                                transforms.Resize((opt.imageSize,opt.imageSize)),
                                #transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((mean_R,mean_G,mean_B), (std_R,std_G,std_B)),
                                ])

trainset = dset.ImageFolder(root='./data/train',
                            transform=transform)
assert trainset



dataloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)


'''
for i, data in enumerate(dataloader, 0):
    if i ==0:
        print(data)
        break
'''


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 512

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d( nz, ngf * 8, 2, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, ngf * 4, 2, 1, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 2, 2, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
#            nn.Conv2d(ngf * 2, ngf, 2, 1, 1, bias=True),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d( ngf*2, nc, 2, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    
    
netG = Generator(ngpu).cuda()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, 256, 2, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(256, 128, 2, 1, 0, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 64, 2, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64, 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(16, 1, 2, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).cuda()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

'''
output = netD(vgg_output)
output.size()
'''

criterion = nn.BCELoss()


real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Use pre-trained VGG16 
model_vgg = models.vgg16(pretrained=True).cuda()

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # train with real
        netD.zero_grad()
        if opt.cuda:
            real_cpu = data[0].cuda()
            real_cpu = Variable(real_cpu)
        else:
            real_cpu = data[0]
    
        batch_size = real_cpu.size(0)
        #label = torch.full((batch_size,), real_label, device=device)
        #label = torch.full((batch_size,), real_label)
        label = Variable(torch.ones(10,)).cuda()
        

        vgg_output=model_vgg.features(real_cpu)
        
        output = netD(vgg_output)
        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean()

        # train with fake
#        noise = torch.randn(batch_size, nz, 1, 1, device=device)
#        fake = netG(noise)
        # 1 means only chanle
        grey = torch.FloatTensor(batch_size,3,opt.imageSize,opt.imageSize).zero_()
        
        # Y' = 0.299 R + 0.587 G + 0.114 B 
        grey[:,0,:,:] = real_cpu[0:batch_size,0,:,:]*0.299 + real_cpu[0:batch_size,1,:,:] * 0.587 + real_cpu[0:batch_size,2,:,:] * 0.114
        grey[:,1,:,:] = grey[:,0,:,:]
        grey[:,2,:,:] = grey[:,0,:,:]


        fake_output=model_vgg.features(grey)

        fake = netG(fake_output)
        
        
        #label.fill_(fake_label)
        label = Variable(torch.zeros(10,)).cuda()
        
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        #label.fill_(real_label)
        label = Variable(torch.ones(10,)).cuda()
        # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean()
        optimizerG.step()

        # D(x) should close to 1  D(G(z))_1 should 1 and D(G(z))_2 should 1
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
#        if i % 100 == 0:
#            vutils.save_image(real_cpu,
#                    '%s/real_samples.png' % opt.outf,
#                    normalize=True)
#            fake = netG(fixed_noise)
#            vutils.save_image(fake.detach(),
#                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
#                    normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))






#plt.imshow(im0c)
#plt.show()





