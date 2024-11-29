import warnings

warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Normalize,
    RandomCrop,
    RandomScale,
)
from albumentations import OneOf, Compose
from collections import OrderedDict
from torch.utils.data import SequentialSampler
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


LABEL_MAP = OrderedDict(
    Background=0, Building=1, Road=2, Water=3, Barren=4, Forest=5, Agricultural=6
)


def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls) * label, new_cls)
    return new_cls


class LoveDA(Dataset):

    num_classes = 7

    def __init__(self, image_dir, mask_dir, transform=None):
        self.rgb_filepath_list = []
        self.mask_filepath_list = []
        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transform

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, "*.tif"))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, "*.png"))

        logger.info(
            "%s -- Dataset images: %d"
            % (os.path.dirname(image_dir), len(rgb_filepath_list))
        )
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        mask_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                mask_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.mask_filepath_list += mask_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.mask_filepath_list) > 0:
            mask = imread(self.mask_filepath_list[idx]).astype(np.uint8) - 1
            if self.transforms is not None:
                image = self.transforms(image)
                mask = self.transforms(mask)

            return {
                "image": image,
                "label": mask,
                "fname": os.path.basename(self.rgb_filepath_list[idx]),
            }
        else:
            if self.transforms is not None:
                image = self.transforms(image)

            return {
                "image": image,
                "fname": os.path.basename(self.rgb_filepath_list[idx]),
            }

    def __len__(self):
        return len(self.rgb_filepath_list)


class LoveDAClassification(LoveDA):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform)

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.mask_filepath_list) > 0:
            mask = imread(self.mask_filepath_list[idx]).astype(np.uint8) - 1
            cls = [1.0 if c in mask else 0.0 for c in LABEL_MAP.values()]
            if self.transforms is not None:
                image = self.transforms(image)
            return {
                "image": image,
                "label": torch.tensor(cls, dtype=torch.float),
                "fname": os.path.basename(self.rgb_filepath_list[idx]),
            }
        else:
            if self.transforms is not None:
                image = self.transforms(image)

            return {
                "image": image,
                "fname": os.path.basename(self.rgb_filepath_list[idx]),
            }


class LoveDAClassificationFog(LoveDAClassification):
    def __init__(self, image_dir, mask_dir, img_haz_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.img_haz_filepath_list = []
        if isinstance(img_haz_dir, list):
            for img_haz_filepath_list in img_haz_dir:
                self.batch_generate(img_haz_filepath_list, None)
        else:
            self.batch_generate(img_haz_dir, None)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        img_haz = imread(self.img_haz_filepath_list[idx])
        if self.transforms is not None:
            img_haz = self.transforms(img_haz)
        data["img_haz"] = img_haz

        return data
