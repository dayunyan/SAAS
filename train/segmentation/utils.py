#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:14:29 2020

@author: user
"""
import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
import cv2
import random
from torchvision import transforms


data_trans = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.RandomCrop((256, 256), padding=25),
    # transforms.RandomAffine(180, (0.1,0.1), (0.8,1), (45,90)),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5)
])
    
def get_batch(batchsize, train_dir, gt_dir, image_size, fusion,itera):
    ids = []
    Inputs = np.zeros([batchsize, 3, image_size, image_size], dtype=np.float32) #input image, 3 channels
    Targets = np.zeros([batchsize, 2, image_size, image_size], dtype=np.float32) #target output:pseudp label binary map
    Smaps = np.zeros([batchsize, 1, image_size, image_size], dtype=np.float32) #pseudo label saliency map
    T = np.zeros([batchsize, 1, image_size, image_size], dtype=np.float32) #confusion heatmap
    for file in os.listdir(gt_dir):
        # print file
        if file.endswith(".tif"):
            # print(file)
            ids.append(str(file))  # list of str
    random.seed(itera)
    ids = random.sample(ids, batchsize)
    # print(ids)
    for i,img_id in enumerate(ids):
        smap = cv2.imread(gt_dir+img_id, cv2.IMREAD_GRAYSCALE)
        smap = cv2.resize(smap[20:-20, 20:-20], (image_size, image_size))
        _, gt = cv2.threshold(smap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #pseudo label binary map
        gt = np.array(gt, dtype=np.float32)/255.0
        image = cv2.imread(train_dir+img_id)
        image = cv2.resize(image[20:-20, 20:-20,:], (image_size, image_size))
        image = image.transpose((2,0,1))
        image = np.array(image, dtype=np.float32) / 255.

        #Data Augmentation
        img_gt = np.concatenate((image, gt[np.newaxis,:]), axis=0)
        temp = data_trans(torch.from_numpy(img_gt[np.newaxis,:]))
        image = temp[0,0:3,:,:].detach().numpy()
        gt = temp[0,3,:,:].detach().numpy()

        Inputs[i, :, :, :] = image
        smap = np.array(smap, dtype=np.float32)/255.
        smap = np.expand_dims(smap,2) #(256,256,1)
        smap = smap.transpose((2,0,1)) #(1,256,256)
        Smaps[i, :, :, :] = smap
        Targets[i, 0, :, :] = gt
        Targets[i, 1, :, :] = 1.-gt
        # cv2.imwrite(f'/home/zjj/xjd/SAAS/pseudo/' + img_id, (gt * 255.).astype(np.uint8))
        
        if fusion:
            th = 0.5
            T[i, 0, :, :] = 2.*np.abs(smap-th) #confusion heatmap
        else:
            T[i, 0, :, :] = 1.

    return Inputs, Targets, Smaps, T

def get_test_batch(batchsize, train_dir, gt_dir, image_size):
    ids = []
    Inputs = np.zeros([batchsize, image_size, image_size, 3],dtype=np.float)
    Targets = np.zeros([batchsize, image_size, image_size, 1] ,dtype=np.float)
    for file in os.listdir(gt_dir):
        # print file
        if file.endswith(".tif"):
            # print(file)
            ids.append(str(file))  # list of str
    ids = ids[0:batchsize]
    # print(ids)
    for i,img_id in enumerate(ids):
        gt = cv2.imread(gt_dir+img_id, cv2.IMREAD_GRAYSCALE)
        gt = np.array(cv2.resize(gt, (image_size, image_size), interpolation=cv2.INTER_NEAREST), np.float)/255.
        image = cv2.imread(train_dir+img_id)
        image = cv2.resize(image, (image_size, image_size))
        # gt = gt[:, :, np.newaxis]
        image = np.array(image, dtype=np.float32) / 255.
        Inputs[i, :, :, :] = image
        Targets[i, :, :, 0] = gt
    return Inputs, Targets

def resize(image,h,w):
    Resize = transforms.Resize(size=(h, w))
    resized_image = Resize(image)
    return resized_image

def save_feature(Feat):
    cha = Feat.shape[1]
    for i in range(cha):
        feature=Feat.detach()[0,i,:,:] #[32,32]
        feature=feature.cpu().numpy()
        feature= 1.0/(1+np.exp(-1*feature)) #[0,1]
        feature=np.round(feature*255) #[0,255]
        cv2.imwrite('./data2/W2F/2020-08-28/temp/feature_%d.png' % i,feature) 
        
def cross_entropy_normal(labels, y_hat, T):
    #beta = tf.reduce_mean(labels[:,:,:,0]) # ratio of foreground
    loss = -torch.mean(torch.squeeze(T)* labels[:,0,:,:] * torch.log(y_hat[:,0,:,:]) +  torch.squeeze(T)* labels[:,1,:,:] * torch.log(y_hat[:,1,:,:]))
    return loss

def loss_deep_supervision(labels, y_hat, batch_size, T, stddev):
    # loss = tf.constant(0)
    loss = []
    loss_sum = 0
    k = 0
    for y in y_hat:
        k = k+1
        tensor_labels = torch.from_numpy(labels)
        tensor_T = torch.from_numpy(T) 
        
        T_matrix_rand = random.gauss(mu=T, sigma=(stddev/k))
        T_matrix_rand = torch.tensor(T_matrix_rand)
        one = torch.ones_like(tensor_T)
        zero = torch.zeros_like(tensor_T)
        T_scaled = torch.where(T_matrix_rand < 0.5, zero, one)

        y1 = (y+1e-5).double()
        temp = cross_entropy_normal(tensor_labels.cuda(), y1.cuda(), T_scaled.cuda())
        loss.append(temp)
        loss_sum += temp
    return loss, loss_sum

def dice_loss(output, label):
    output0 = output[:, 0, :, :]
    output1 = output[:, 1, :, :]
    intersection0 = output0 * label[:,0,:,:]
    intersection1 = output1 * label[:,1,:,:]
    DSC0 = (2 * torch.abs(torch.sum(intersection0)) + 1) / (torch.abs(torch.sum(output0)) + torch.sum(label[:,0,:,:]) + 1)
    DSC1 = (2 * torch.abs(torch.sum(intersection1)) + 1) / (torch.abs(torch.sum(output1)) + torch.sum(label[:,1,:,:]) + 1)
    loss = 1 - (DSC0 + DSC1) / 2
    # print('dice:', loss.mean())

    return loss
