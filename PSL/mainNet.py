from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
# from visdom import Visdom
import numpy as np
import time
import argparse
from geoconfig import config, args
from utils import get_batch,get_test_batch,loss_deep_supervision
from models import Feedback_saliency_generator
import matplotlib.pyplot as plt
import cv2
import os
# viz = Visdom(env='PSL_mj')
writer = SummaryWriter('./log'+'/train')


full_data_flag = args.full_data
if full_data_flag==0:
    print('part data')
    gt_dir = config.TRAIN.gt_dir
    train_dir  = config.TRAIN.train_dir
else:
    print('full data')
    gt_dir = config.TRAIN.full_gt_dir
    train_dir  = config.TRAIN.full_train_dir

model_dir = config.TRAIN.model_dir
test_dir = config.TRAIN.test_dir
test_gt_dir  = config.TRAIN.test_gt_dir 
save_dir = config.TRAIN.save_dir
image_size = config.TRAIN.image_size
batchsize = config.TRAIN.batch_size
epoch = config.TRAIN.n_epoch
iteration = config.TRAIN.iteration
D = config.TRAIN.D
C = config.TRAIN.C

std_value = args.std_value
fusion = args.fusion
feature_num = args.features
Tini = args.Tini
Dnum = args.D
refine = args.refine

FBSN = Feedback_saliency_generator(features = feature_num, D = Dnum, T_int = Tini,if_refine = refine)
device = config.device
# if torch.cuda.is_available():
FBSN = FBSN.to(device)
opt_FBSN = optim.Adam(FBSN.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(opt_FBSN, mode='min', factor=0.5, patience=20, verbose=True)


def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+'best weights.pth')
    
def load_model(step, model, path):
    #  restore models
    model.load_state_dict(torch.load(path+str(step)+'.pth'))


start_time = time.time()
time_p, Loss = [], []
# line = viz.line(np.arange(epoch))
k = 0
#step = 49
#load_model(step,FBSN, model_dir)
min_loss = 100000
for step in range(epoch):
    loss_list = []
    for itera in range(iteration):
        (train_x, train_y, train_Smaps, train_T) = get_batch(batchsize, train_dir, gt_dir, image_size, fusion,
                                                             step * epoch + itera)
        #test_image_batch, test_gt_batch = get_test_batch(batchsize, test_dir, test_gt_dir, image_size)

        train_x_tensor = torch.from_numpy(train_x)
        train_x_tensor = Variable(torch.tensor(train_x_tensor, dtype=torch.float32))
        train_x_tensor = train_x_tensor.to(device)
        output = FBSN(train_x_tensor)
        opt_FBSN.zero_grad()
        loss_temp, loss = loss_deep_supervision(train_y, output, batchsize, train_T, std_value)
        loss_list.append(loss.data.mean())
        loss.backward(retain_graph=True)
        opt_FBSN.step()
        
        writer.add_scalar('Train/Loss', loss, itera+step*iteration)

#        if (itera + 1) % 2 == 0 and itera != 0:
#            pred = loss
#            k = k + 1
#            localtime = time.asctime(time.localtime(time.time()))
#            print(localtime)
#            print('model:epoch:%d, iter:%d,  loss:%.4f' % (step, itera, pred))

    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    print('[%d/%d]: Loss_train:%.4f'
          % (step,
             epoch,
             torch.mean(torch.stack(loss_list))))

    #draw curves
    time_p.append(step+1)
    #time_p.append(time.time() - start_time)
    Loss.append(torch.mean(torch.stack(loss_list)).item())
    scheduler.step(Loss[step])
#     viz.line(X=np.array(time_p),
#              Y=np.array(Loss),
#              win=line,
#              opts=dict(title='train loss'))

    if (step + 1) % 10 == 0 and step != 0:
        for i in range(batchsize):
            a = (train_x[i, :, :, :]).transpose(1, 2, 0)
            cv2.imwrite(save_dir + 'origin_%d.png' % i, (a * 255.).astype(np.uint8))
            out = output[-1].cpu()
            out = out.detach().numpy() #(16, 2, 256, 256)
            out = np.array(out)
            sa = np.squeeze(out)[:, 0, :, :] #(16, 256, 256)
            cv2.imwrite(save_dir+'pred_%d_epoch_%d.png' % (i, step), (sa[i, :, :]*255.).astype(np.uint8))
#    if (step + 1) % 10 == 0 and step != 0:
#        save_model(step, FBSN, model_dir)
#        print("Save Model")
    temp = torch.mean(torch.stack(loss_list))
    if temp < min_loss:
        min_loss = temp + 0.0005
        print("save model in step %d" %(40 * k + step))
        save_model(FBSN, model_dir)

writer.close()
