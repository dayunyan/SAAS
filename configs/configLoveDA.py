import os
from easydict import EasyDict as edict

config = edict()
config.NAME = "love"

config.weight = 1024
config.height = 1024
config.batch_size = 32
config.num_epoch = 5
config.num_workers = 0
config.num_classes = 7
config.logfile_dir = "log/"

root = "/home/zjj/xjd/"
dataset_path = os.path.join(root, "datasets/LoveDA")
psm_path = os.path.join(root, "psm/LoveDA")
model_path = os.path.join(root, "SAAS/models/save/love")

config.train_mask_dir = [
    f"{dataset_path}/Train/Rural/masks_png",
    f"{dataset_path}/Train/Urban/masks_png",
]
config.val_mask_dir = [
    f"{dataset_path}/Val/Rural/masks_png",
    f"{dataset_path}/Val/Urban/masks_png",
]

config.CLEAR = edict()
config.CLEAR.name = "clear"
config.CLEAR.train_dir = [
    f"{dataset_path}/Train/Rural/images_png",
    f"{dataset_path}/Train/Urban/images_png",
]
config.CLEAR.val_dir = [
    f"{dataset_path}/Val/Rural/images_png",
    f"{dataset_path}/Val/Urban/images_png",
]
config.CLEAR.test_dir = [
    f"{dataset_path}/Test/Rural/images_png",
    f"{dataset_path}/Test/Urban/images_png",
]

config.CLEAR.train_psm = f"{psm_path}/train_psm"
config.CLEAR.val_psm = f"{psm_path}/val_psm"
config.CLEAR.test_psm = f"{psm_path}/test_psm"

config.CLEAR.model_path = f"{model_path}/model_resnet50_best.pth"
