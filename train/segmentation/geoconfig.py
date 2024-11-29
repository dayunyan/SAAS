#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:10:51 2020

@author: user
"""
import argparse
import torch
from easydict import EasyDict as edict
import sys

sys.path.append("/home/zjj/xjd/SAAS/")
import json


p = argparse.ArgumentParser()
p.add_argument(
    "--mode",
    type=str,
    default="clear",
    choices=["clear", "mild", "severe", "nonuniform"],
    help="type of haze",
)
p.add_argument("--data_name", type=str, choices=["geo", "spot"])
p.add_argument("--device", type=str, default="cpu")
p.add_argument("--fusion", type=int, default=1)
p.add_argument("--std_value", type=float, default=0.1)
p.add_argument("--D", type=int, default=1)
p.add_argument("--Tini", type=int, default=3)
p.add_argument("--features", type=int, default=32)
p.add_argument("--full_data", type=int, default=1)
p.add_argument("--refine", type=int, default=1)
p.add_argument("--learning_rate", type=float, default=0.001)
p.add_argument("--cam", type=str, default="cam")
p.add_argument(
    "--dehaze", type=str, default="none", choices=["none", "dcp", "aod", "ffa"]
)
p.add_argument(
    "--wo", type=str, default="none", choices=["none", "dcfl", "darm", "dpo"]
)
p.add_argument("--net", type=str, default="fsan", choices=["fsan", "refinenet"])
args = p.parse_args()

if args.data_name == "geo":
    from configs.config import config as cg
else:
    from configs.configSPOT import config as cg

if args.mode == "clear":
    args.mode = cg.CLEAR
elif args.mode == "mild":
    args.mode = cg.HAZE
elif args.mode == "severe":
    args.mode = cg.DEEPHAZE
else:
    args.mode = cg.NONUNIHAZE

MODE = args.mode

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16  # 8
config.TRAIN.image_size = 256
config.TRAIN.n_epoch = 100
config.TRAIN.iteration = 64  # 64
config.TRAIN.D = 4
config.TRAIN.C = 3
config.TRAIN.if_fusion = 1
config.TRAIN.feature_num = 32
config.TRAIN.std_value = 0.5

config.device = torch.device(args.device)

folder_name = "2021-11-09 geo/"

## train set location
# config.TRAIN.gt_dir = './train/smap/'
# config.TRAIN.train_dir = './train/geo/'
# config.TRAIN.test_dir ='/test/img/'
# config.TRAIN.test_gt_dir = './test/gt/'
# config.TRAIN.save_dir = './train_pred/'
# config.TRAIN.model_dir = './state_dict/'
config.TRAIN.edge_dir = "/home/deeplearning2/lyn/VAE-MSGAN/datasets/island_texture/"
config.TRAIN.test_edge_dir = "/home/deeplearning2/lyn/W2F_island/test/texture/"

# config.TRAIN.full_gt_dir = './train/smap_full/'#'/data2/W2F/dataset/train/full_data/smap/'
# config.TRAIN.full_train_dir = './train/island_full/'#'/data2/W2F/dataset/train/full_data/residential/'

if cg.NAME == "geo":
    config.TRAIN.test_gt_dir = "/home/zjj/xjd/datasets/geoeye-1/geo_test/gt/"
    config.TRAIN.save_dir = "/home/zjj/xjd/SAAS/train/segmentation/pred/geo/train_pred/"
else:
    config.TRAIN.test_gt_dir = "/home/zjj/xjd/datasets/spot5/spot5_test/gt/"
    config.TRAIN.save_dir = (
        "/home/zjj/xjd/SAAS/train/segmentation/pred/spot/train_pred/"
    )

# config.TRAIN.gt_dir = '../dataset/geoeye-1/train_psm/'
# config.TRAIN.train_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_train/train/residential/'
# config.TRAIN.test_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_test/img/'
# config.TRAIN.test_gt_dir = '/root/autodl-tmp/xjd/data/geoeye-1/geo_test/gt/'
# config.TRAIN.save_dir = '../pred/geo/train_pred/'
# config.TRAIN.model_dir = './state_dict/'

# config.TRAIN.full_gt_dir = '../dataset/geoeye-1/smap_full/'
# config.TRAIN.full_train_dir = '/root/autodl-tmp/xjd/data/geoeye-1/full_residential/'

if args.dehaze == "none":
    if args.wo == "none":
        #######################################################################################
        config.TRAIN.gt_dir = MODE.train_psm
        config.TRAIN.train_dir = MODE.train_dir + "residential/"
        config.TRAIN.test_dir = MODE.test_dir

        config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/state_dict_{MODE.name}_{args.cam}/"

        config.TRAIN.full_gt_dir = MODE.full_train_gt
        config.TRAIN.full_train_dir = MODE.full_train

    elif args.wo == "dcfl":
        #######################################################################################
        config.TRAIN.gt_dir = MODE.train_nonconsist_psm
        config.TRAIN.train_dir = MODE.train_dir + "residential/"
        config.TRAIN.test_dir = MODE.test_dir

        config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/woDCFL_state_dict_{MODE.name}_{args.cam}/"

        config.TRAIN.full_gt_dir = MODE.full_nonconsist_train_gt
        config.TRAIN.full_train_dir = MODE.full_nonconsist_train

    #######################################################################################
    # config.TRAIN.gt_dir = MODE.train_pcares_perloss_psm
    # config.TRAIN.train_dir = MODE.train_dir+'residential/'
    # config.TRAIN.test_dir = MODE.test_dir

    # config.TRAIN.model_dir = f'./{cg.NAME}/pcares_perloss_state_dict_'+MODE.name+'/'

    # config.TRAIN.full_gt_dir = MODE.full_pcares_perloss_train_gt
    # config.TRAIN.full_train_dir = MODE.full_pcares_perloss_train

    elif args.wo == "darm":
        #######################################################################################
        config.TRAIN.gt_dir = MODE.train_pcares_woAtt_perloss_psm
        config.TRAIN.train_dir = MODE.train_dir + "residential/"
        config.TRAIN.test_dir = MODE.test_dir

        config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/woDARM_state_dict_{MODE.name}_{args.cam}/"

        config.TRAIN.full_gt_dir = MODE.full_pcares_woAtt_perloss_train_gt
        config.TRAIN.full_train_dir = MODE.full_pcares_woAtt_perloss_train

    elif args.wo == "dpo":
        #######################################################################################
        config.TRAIN.gt_dir = MODE.train_pcares_woPerloss_psm
        config.TRAIN.train_dir = MODE.train_dir + "residential/"
        config.TRAIN.test_dir = MODE.test_dir

        config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/woDPO_state_dict_{MODE.name}_{args.cam}/"

        config.TRAIN.full_gt_dir = MODE.full_pcares_woPerloss_train_gt
        config.TRAIN.full_train_dir = MODE.full_pcares_woPerloss_train

elif args.dehaze == "dcp":
    #######################################################################################
    config.TRAIN.gt_dir = MODE.train_dehaze_psm
    config.TRAIN.train_dir = MODE.train_dehaze_dir + "residential/"
    config.TRAIN.test_dir = MODE.test_dehaze_dir

    config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/dehaze_state_dict_{MODE.name}_{args.cam}/"

    config.TRAIN.full_gt_dir = MODE.full_dehaze_train_gt
    config.TRAIN.full_train_dir = MODE.full_dehaze_train

elif args.dehaze == "aod":
    #######################################################################################
    config.TRAIN.gt_dir = MODE.train_AODdehaze_psm
    config.TRAIN.train_dir = MODE.train_AODdehaze_dir + "residential/"
    config.TRAIN.test_dir = MODE.test_AODdehaze_dir

    config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/AODdehaze_state_dict_{MODE.name}_{args.cam}/"

    config.TRAIN.full_gt_dir = MODE.full_AODdehaze_train_gt
    config.TRAIN.full_train_dir = MODE.full_AODdehaze_train

elif args.dehaze == "ffa":
    #######################################################################################
    config.TRAIN.gt_dir = MODE.train_FFAdehaze_psm
    config.TRAIN.train_dir = MODE.train_FFAdehaze_dir + "residential/"
    config.TRAIN.test_dir = MODE.test_FFAdehaze_dir

    config.TRAIN.model_dir = f"/home/zjj/xjd/SAAS/train/segmentation/save/{args.net}/{cg.NAME}/FFAdehaze_state_dict_{MODE.name}_{args.cam}/"

    config.TRAIN.full_gt_dir = MODE.full_FFAdehaze_train_gt
    config.TRAIN.full_train_dir = MODE.full_FFAdehaze_train

#######################################################################################
# config.TRAIN.gt_dir = '../'+MODE.train_APFFdehaze_psm
# config.TRAIN.train_dir = '../'+MODE.train_APFFdehaze_dir+'residential/'
# config.TRAIN.test_dir = '../'+MODE.test_APFFdehaze_dir

# config.TRAIN.model_dir = f'./{cg.NAME}/APFFdehaze_state_dict_'+MODE.name+'/'

# config.TRAIN.full_gt_dir = '../'+MODE.full_APFFdehaze_train_gt
# config.TRAIN.full_train_dir = '../'+MODE.full_APFFdehaze_train

#######################################################################################
config.VALID = edict()
# config.VALID.save_dir = './val_pred/'

# config.VALID.val_dir = '/media/mj/W2F/geo/2019-12-04/dataset/validation/residential/'
# config.VALID.gt_dir = '/media/mj/W2F/geo/2019-12-04/dataset/validation/gt/'
# config.VALID.save_dir = '../pred_haze/geo/val_pred/'
config.VALID.save_dir = (
    f"/home/zjj/xjd/SAAS/train/segmentation/pred/{cg.NAME}/val_pred/"
)
# config.VALID.model_dir = './' + folder_name + '/modelCorrectNet'


def log_config(filename, cfg):
    with open(filename, "w") as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
