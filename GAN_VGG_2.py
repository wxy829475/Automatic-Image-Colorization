#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:31:29 2018

@author: tinghao
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
import torch.nn.functional as F

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
parser.add_argument('--lam', type=float, default = 100, help='regularization coeff')


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
        
                # up-sampling and conv

        self.up1 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'), #512x16x16
                                 nn.Conv2d(512,256,3,padding = 1), #256x16x16
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2) ,                                
                                 )        
               # up-sampling and conv
 
        self.up2 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'), #256x32x32
                                 nn.Conv2d(256,128,3,padding = 1), #128x32x32
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2) ,                                
                                 )    
                # up-sampling and conv

        self.up3 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'), #128x64x64
                                 nn.Conv2d(128,64,3,padding = 1), #64x64x64
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2) ,                                
                                 )    
        # up-sampling and conv
        
        self.up4 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'), #64x128x128
                                 nn.Conv2d(64,32,3,padding = 1), #32x128x128
                                 nn.BatchNorm2d(32),
                                 nn.LeakyReLU(0.2) ,                                
                                 )    
        ####
        self.conv5 = nn.Sequential(nn.Conv2d(32,64,3,padding = 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2),#64x128x128
                                   )
        self.pool6 =  nn.MaxPool2d((2,2)) #64x64x64

        # concat layer 3
        self.conv7 = nn.Sequential(nn.Conv2d(128,64,3,padding = 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2),#64x64x64
                                   )

        #### 
        self.conv8 = nn.Sequential(nn.Conv2d(64,128,3,padding = 1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2),#64x64x64
                                   )
        self.pool9 =  nn.MaxPool2d((2,2)) #64x32x32
        
        
        # concat layer 2
        self.conv10 = nn.Sequential(nn.Conv2d(256,128,3,padding = 1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2),#128x32x32
                                   )    
    
        ####
        self.conv11 = nn.Sequential(nn.Conv2d(128,256,3,padding = 1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2),#256x32x32
                                   )
        self.pool12 =  nn.MaxPool2d((2,2)) #256x16x16

        # concat layer 1
        self.conv13 = nn.Sequential(nn.Conv2d(512,256,3,padding = 1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2),#256x16x16
                                   )
        
        ####
        self.conv14 = nn.Sequential(nn.Conv2d(256,512,3,padding = 1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2),#512x16x16
                                   )
        self.pool15 =  nn.MaxPool2d((2,2)) #512x8x8

        # concat layer 0
        self.conv16 = nn.Sequential(nn.Conv2d(1024,512,3,padding = 1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2),#512x8x8
                                   )



    def forward(self, input):
        x0 = input
        u1 = self.up1(x0)
        #x = self.pool1(c1)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        print(u4.size())
        
        
        c5 = self.conv5(u4)
        p6 = self.pool6(c5)
        print(p6.size())
        temp_cov = torch.cat([p6,u3],dim = 1)
        c7 = self.conv7(temp_cov)
        print(c7.size())
        
        c8 = self.conv8(c7)
        p9 = self.pool9(c8)
        print(p9.size())
        temp_cov = torch.cat([p9,u2],dim = 1)
        c10 = self.conv10(temp_cov)
        print(c10.size())

        c11 = self.conv11(c10)
        p12 = self.pool12(c11)
        print(p12.size())
        temp_cov = torch.cat([p12,u1],dim = 1)
        c13 = self.conv13(temp_cov)
        print(c13.size())        
        
        c14 = self.conv14(c13)
        p15 = self.pool15(c14)
        print(p15.size())
        temp_cov = torch.cat([p15,x0],dim = 1)
        x = self.conv16(temp_cov)
        print(x.size())
        
        
        
        
        
        
        if input.is_cuda and self.ngpu >= 1:
            output = x.cuda()
        else:
            output = x
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
                                  nn.Conv2d(512, 256, 3, padding = 1,bias=True),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  
                                  nn.Conv2d(256, 128, 3, padding = 1, bias=True),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.MaxPool2d((2,2)), #128x4x4
                                  
                                  
                                  nn.Conv2d(128, 64, 3,padding = 1, bias=True),#256x16x16
                                  nn.LeakyReLU(0.2, inplace=True),
                                  #128 *4*4
                                  
                                  nn.Conv2d(64, 16, 2, padding = 1, bias=True), #512x8x8
                                  nn.LeakyReLU(0.2, inplace=True),
                                  # 64 x 3 x3
                                  
                                  nn.Conv2d(16,1,3, bias=True),
                                  nn.LeakyReLU(0.2, inplace=True) #512x4x4
                                  
                                  
                                  )
    
    def forward(self, input):
        x = input
        x = self.main(x)
        
        x = F.sigmoid(x)
        
        if input.is_cuda and self.ngpu >= 1:
            output = x.cuda()
        else:
            output = x
        
        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).cuda()

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


criterion = nn.BCELoss()
criterion_translation = torch.nn.L1Loss()

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Use pre-trained VGG16 
model_vgg = models.vgg16(pretrained=True)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # train with real
        netD.zero_grad()
        
        real_cpu = data[0]
        real_cpu = Variable(real_cpu).cuda()
        print(real_cpu.size())
        
        batch_size = real_cpu.size(0)
        
        vgg_real=model_vgg.features(real_cpu)

        
        #label = torch.full((batch_size,), real_label, device=device)
        #label = torch.full((batch_size,), real_label)
        
        #label = Variable(torch.ones(10,)).cuda()
        
        output = netD(vgg_real).cuda()
    
        errD_real = criterion(output,Variable(torch.ones(output.size(),)).cuda())
        errD_real.backward()
        D_x = output.mean()
        
        # train with fake
        #        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        #        fake = netG(noise)
        # 1 means only chanle
        grey = torch.FloatTensor(batch_size,1,opt.imageSize,opt.imageSize).zero_()
        
        # Y' = 0.299 R + 0.587 G + 0.114 B
        grey[:,0,:,:] = real_cpu.data[0:batch_size,0,:,:]*0.299 + real_cpu.data[0:batch_size,1,:,:] * 0.587 + real_cpu.data[0:batch_size,2,:,:] * 0.114
        grey[:,1,:,:] = grey[:,0,:,:]
        grey[:,2,:,:] = grey[:,0,:,:]
        
        grey = Variable(grey)
  
        
        vgg_fake=model_vgg.features(grey).cuba()
        
        fake = netG(vgg_fake).cuda()

        
        #label.fill_(fake_label)
    
        #label = Variable(torch.zeros(10,)).cuda()

        
        output2 = netD(fake.detach()).cuda()
        errD_fake = criterion(output2, Variable(torch.zeros(output2.size(),)).cuda())
        errD_fake.backward()
        D_G_z1 = output2.mean()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        #label.fill_(real_label)
    
        #label = Variable(torch.ones(10,)).cuda()

        # fake labels are real for generator cost
        output3 = netD(fake).cuda()
        
        # Tinghao Updates
        loss_trans = criterion_translation(fake, vgg_real)
        
        
        #errG = criterion(output3, label)+ opt.lam*criterion_translation(fake, real_cpu)
        errG = criterion(output3, Variable(torch.ones(output3.size(),)).cuda())+loss_trans*opt.lam
        
  
        errG.backward()
        D_G_z2 = output3.mean()
        optimizerG.step()
        
        # D(x) should close to 1  D(G(z))_1 should 1 and D(G(z))_2 should 1
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG.data, D_x, D_G_z1, D_G_z2))

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






