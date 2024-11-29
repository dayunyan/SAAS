# -*- coding: UTF-8 -*-
import torch
from utils import *
from models import *
from geoconfig import config, args
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import cv2
import os
from shutil import copyfile
import argparse
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import classification_report
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def model_dir_change(model_dir, test_epoch):
    fw = open(model_dir + '/checkpoint', 'a+')
    new_stus = ''
    fw.seek(0)
    test_epoch = test_epoch
    for s in fw:
        if not s.find('all'):
            stu = s
        else:
            stu = 'model_checkpoint_path: "model-' + str(test_epoch) + '.ckpt"' + '\n'
        new_stus = new_stus + stu
        # print new_stus
        # break
    fw.seek(0)
    fw.truncate()
    fw.write(new_stus)


# p = argparse.ArgumentParser()
# p.add_argument('-model_dir', default='./state_dict/', type=str)
# args = p.parse_args()

val_dir = config.TRAIN.test_dir
gt_dir = config.TRAIN.test_gt_dir

D_num = 1
T_num = 3
refine = 1

val_dir = config.TRAIN.test_dir
gt_dir = config.TRAIN.test_gt_dir
save_dir = config.VALID.save_dir
model_dir = config.TRAIN.model_dir
feature_num = config.TRAIN.feature_num
image_size = config.TRAIN.image_size

device = config.device

FBSN = Feedback_saliency_generator(features = feature_num, D = D_num, T_int = T_num,if_refine = refine).to(device)
print('# model parameters:', sum(param.numel()*param.element_size() for param in FBSN.parameters())/1024/1024)

def load_model(model, path):
    #  restore models
    model.load_state_dict(torch.load(path+'best weights.pth'))

load_model(FBSN, model_dir)
#model_dir_change(model_dir, test_epoch)
#copyfile('./geoconfig.py', save_dir + '/geoconfig.py')
print('Loading Model...')
    
def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    #return img.float().div(255).unsqueeze(0)  # 255也可以改为25
    return img.unsqueeze(0)
    
image_num = 0
prec_sum = 0
recall_sum = 0
fscore_sum = 0
auc_sum = 0


for file in os.listdir(val_dir):
    # print file
    if file.endswith(".tif"):
        image_num = image_num + 1
        im = np.array(cv2.resize(cv2.imread(val_dir + file), (image_size, image_size)), np.float32) / 255.
        print(gt_dir + file)
        gt = np.array(cv2.resize(cv2.imread(gt_dir + file, cv2.IMREAD_GRAYSCALE), (image_size, image_size),
                                interpolation=cv2.INTER_NEAREST), np.float32) / 255.                
        
        test_x_tensor = toTensor(im)
        #print(test_x_tensor.shape) #[1, 3, 256, 256]
        test_x_tensor = Variable(torch.as_tensor(test_x_tensor, dtype=torch.float32))
        test_x_tensor = test_x_tensor.to(device)
        
        start = time.perf_counter()
        output = FBSN(test_x_tensor)
        end = time.perf_counter()

        sa = output[-1].cpu() #[1, 2, 256, 256]
        sa = sa.detach().numpy()  # (1, 2, 256, 256)
        sa = np.array(sa)
        sa = np.squeeze(sa)[0, :, :] #(256,256)
        sa_color = np.expand_dims(sa, axis=2)
        sa_color = np.tile(sa_color, [1, 1, 3])
        im_color = im * sa_color
        sa_image = sa
        sa = np.reshape(sa, (image_size * image_size, 1))
        gt = np.reshape(gt, (image_size * image_size, 1))
        sgt = (255*(sa_image-sa_image.min())/(sa_image.max()-sa_image.min())).astype(np.uint8)
        thr, sgt = cv2.threshold(sgt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sgt = np.reshape(sgt/255., (image_size * image_size, 1))
        fpr, tpr, thresholds = roc_curve(gt.astype('int'), sgt.astype('int'), pos_label=1)
        auc_sum += auc(fpr, tpr)
        print(thr)
#         print(gt)
        print('auc_temp:',auc(fpr, tpr))
        cv2.imwrite(save_dir + file, (sa_image * 255.).astype(np.uint8))
        cv2.imwrite(save_dir + 'roi' + file, (im_color * 255.).astype(np.uint8))
        
        res_met = classification_report(gt.astype('int'),sgt.astype('int'),target_names=['fore','back'],output_dict=True)
        prec_sum += res_met['fore']['precision']
        recall_sum += res_met['fore']['recall']
        fscore_sum += res_met['fore']['f1-score']
print(f'auc: {auc_sum / image_num}; prec_sum: {prec_sum/image_num}; recall_sum: {recall_sum/image_num}; fscore_sum: {fscore_sum/image_num}')
print(f'time:{end-start}')