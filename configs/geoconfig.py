#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:10:51 2020

@author: user
"""
from easydict import EasyDict as edict
import sys 
sys.path.append('../') 
from configs.config import config as cg
# from configs.configSPOT import config as cg
import json

MODE = cg.HAZE

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16 #8
config.TRAIN.image_size = 256
config.TRAIN.n_epoch = 500
config.TRAIN.iteration = 8
config.TRAIN.D = 4
config.TRAIN.C = 3
config.TRAIN.if_fusion = 1
config.TRAIN.feature_num = 32
config.TRAIN.std_value = 0.5

folder_name = '2021-11-09 geo/'

## train set location
# config.TRAIN.gt_dir = './train/smap/'
# config.TRAIN.train_dir = './train/geo/'
# config.TRAIN.test_dir ='/test/img/'
# config.TRAIN.test_gt_dir = './test/gt/'
# config.TRAIN.save_dir = './train_pred/'
# config.TRAIN.model_dir = './state_dict/'
config.TRAIN.edge_dir = '/home/deeplearning2/lyn/VAE-MSGAN/datasets/island_texture/'
config.TRAIN.test_edge_dir = '/home/deeplearning2/lyn/W2F_island/test/texture/'

# config.TRAIN.full_gt_dir = './train/smap_full/'#'/data2/W2F/dataset/train/full_data/smap/'
# config.TRAIN.full_train_dir = './train/island_full/'#'/data2/W2F/dataset/train/full_data/residential/'

if cg.NAME=='geo':
    config.TRAIN.test_gt_dir = '/home/zjj/xjd/datasets/geoeye-1/geo_test/gt/'
    config.TRAIN.save_dir = 'pred_haze/geo/train_pred/'
else:
    config.TRAIN.test_gt_dir = '/root/autodl-tmp/xjd/data/spot5/spot5_test/gt/'
    config.TRAIN.save_dir = 'pred_haze/spot5/train_pred/'

# config.TRAIN.gt_dir = '../dataset/geoeye-1/train_psm/'
# config.TRAIN.train_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_train/train/residential/'
# config.TRAIN.test_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_test/img/'
# config.TRAIN.test_gt_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_test/gt/'
# config.TRAIN.save_dir = '../pred/geo/train_pred/'
# config.TRAIN.model_dir = './state_dict/'

# config.TRAIN.full_gt_dir = '../dataset/geoeye-1/smap_full/'
# config.TRAIN.full_train_dir = '/root/autodl-tmp/xjd/data/geoeye-1/full_residential/'

#######################################################################################
# config.TRAIN.gt_dir = MODE.train_psm
# config.TRAIN.train_dir = MODE.train_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_train_gt
# config.TRAIN.full_train_dir = MODE.full_train

#######################################################################################
# config.TRAIN.gt_dir = '../'+MODE.train_perloss_psm
# config.TRAIN.train_dir = '../'+MODE.train_dir+'residential/'
# config.TRAIN.test_dir = '../'+MODE.test_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/perloss_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = '../'+MODE.full_perloss_train_gt
# config.TRAIN.full_train_dir = '../'+MODE.full_perloss_train

####################################################################################### 
# config.TRAIN.gt_dir = MODE.train_pcares_perloss_psm
# config.TRAIN.train_dir = MODE.train_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/pcares_perloss_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_pcares_perloss_train_gt
# config.TRAIN.full_train_dir = MODE.full_pcares_perloss_train

#######################################################################################
# config.TRAIN.gt_dir = MODE.train_pcares_woAtt_perloss_psm
# config.TRAIN.train_dir = MODE.train_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/pcares_woAtt_perloss_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_pcares_woAtt_perloss_train_gt
# config.TRAIN.full_train_dir = MODE.full_pcares_woAtt_perloss_train

#######################################################################################
config.TRAIN.gt_dir = MODE.train_pcares_woPerloss_psm
config.TRAIN.train_dir = MODE.train_dir+'residential/'
config.TRAIN.test_dir = MODE.test_dir

config.TRAIN.model_dir = f'./{cg.NAME}/pcares_woPerloss_state_dict_'+MODE.name+'/'

config.TRAIN.full_gt_dir = MODE.full_pcares_woPerloss_train_gt
config.TRAIN.full_train_dir = MODE.full_pcares_woPerloss_train

#######################################################################################
# config.TRAIN.gt_dir = MODE.train_dehaze_psm
# config.TRAIN.train_dir = MODE.train_dehaze_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_dehaze_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/dehaze_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_dehaze_train_gt
# config.TRAIN.full_train_dir = MODE.full_dehaze_train

#######################################################################################
# config.TRAIN.gt_dir = MODE.train_AODdehaze_psm
# config.TRAIN.train_dir = MODE.train_AODdehaze_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_AODdehaze_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/AODdehaze_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_AODdehaze_train_gt
# config.TRAIN.full_train_dir = MODE.full_AODdehaze_train

#######################################################################################
# config.TRAIN.gt_dir = MODE.train_FFAdehaze_psm
# config.TRAIN.train_dir = MODE.train_FFAdehaze_dir+'residential/'
# config.TRAIN.test_dir = MODE.test_FFAdehaze_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/FFAdehaze_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = MODE.full_FFAdehaze_train_gt
# config.TRAIN.full_train_dir = MODE.full_FFAdehaze_train

#######################################################################################
# config.TRAIN.gt_dir = '../'+MODE.train_APFFdehaze_psm
# config.TRAIN.train_dir = '../'+MODE.train_APFFdehaze_dir+'residential/'
# config.TRAIN.test_dir = '../'+MODE.test_APFFdehaze_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/APFFdehaze_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = '../'+MODE.full_APFFdehaze_train_gt
# config.TRAIN.full_train_dir = '../'+MODE.full_APFFdehaze_train

#######################################################################################
config.VALID = edict()
#config.VALID.save_dir = './val_pred/'

#config.VALID.val_dir = '/media/mj/W2F/geo/2019-12-04/dataset/validation/residential/'
#config.VALID.gt_dir = '/media/mj/W2F/geo/2019-12-04/dataset/validation/gt/'
# config.VALID.save_dir = '../pred_haze/geo/val_pred/'
config.VALID.save_dir = f'pred_haze/{cg.NAME}/val_pred/'
#config.VALID.model_dir = './' + folder_name + '/modelCorrectNet'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
