#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:48:08 2020

@author: user
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import save_feature


class conv(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride_val, padding_val):
        super(conv, self).__init__()
        # input parameters
        self.feat = nn.Sequential()
        self.feat.add_module("conv_1", nn.Conv2d(in_channels, out_channels, kernel_size, stride_val, padding_val))
        self.feat.add_module("relu_1", nn.ReLU())

    def forward(self, x):
        output = self.feat(x)
        return output


class HDC_block_Conv(nn.Module):
    def __init__(self, features=64, D=1):
        super(HDC_block_Conv, self).__init__()
        # input parameters
        stride_val = 1
        self.D = D
        self.features = features
        self.conv1 = nn.Conv2d(2*features, features, 3, stride_val, 1)
        self.conv2 = nn.Conv2d(features, features, 3, stride_val, 1)
        self.conv3 = nn.Conv2d(features, features, 3, stride_val, 1)

    def forward(self, F_in, F_out):
        n = torch.cat((F_in, F_out), 1)
        innodes = self.conv1(n)
        for i in range(self.D):
            innodes = self.conv2(innodes)
            n = self.conv3(innodes)
        return n


class gen_sa(nn.Module):
    def __init__(self, feat=64):
        super(gen_sa, self).__init__()
        # input parameters
        stride_val = 1
        self.feat = feat
        self.conv1 = nn.Conv2d(feat, feat, 3, stride = stride_val, padding = 1)
        self.conv2 = nn.Conv2d(feat, feat, 3, stride = stride_val, padding = 1)
        self.conv3 = nn.Conv2d(feat, 2, 1, stride = stride_val, padding = 0)

    def forward(self, F_in):
     
        n = self.conv1(F_in)
        n = self.conv2(n)
        n = self.conv3(n)
        return n


class Feedback_saliency_generator(nn.Module):
    def __init__(self, features=64, D=2, T_int=3, if_refine=1):
        super(Feedback_saliency_generator, self).__init__()
        # input parameters
        stride_value = 1
        self.features = features
        self.D = D
        self.T_int = T_int
        self.if_refine = if_refine
        self.conv_n = conv(kernel_size=3, in_channels=3, out_channels=self.features, stride_val=stride_value, padding_val=1)
        self.conv_F_in = conv(kernel_size=3, in_channels =self.features, out_channels=self.features, stride_val=stride_value, padding_val=1)
        self.HDC = HDC_block_Conv(features=self.features, D=D)
        self.gen_SA = gen_sa(feat=self.features)

    def forward(self, input_image):
        n = self.conv_n(input_image)
        F_in = self.conv_F_in(n)
        FB_out = 0
        # recurrent blocks
        #primary_saliency_map = Variable(torch.zeros([16, self.T_int, 256, 256])).cuda()
        outs = []
        for i in range(self.T_int):
            #print(i)
            F_out = FB_out if i > 0 else F_in
            FB_out = self.HDC(F_in, F_out)
            temp = self.gen_SA(FB_out)
            outs.append(F.softmax(temp, dim=1))
            #primary_saliency_map[:, i, :, :] = temp
        save_feature(FB_out)
        if self.if_refine == 1:
            coenfficient = Variable(torch.zeros([self.T_int])).cuda() + 1. / 3.
            final_smap = 0
            for j in range(self.T_int):
                final_smap = final_smap + coenfficient[j] * outs[j]
            outs.append(F.softmax(final_smap, dim=1))
        return outs
