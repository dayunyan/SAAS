import sys, os

sys.path.append("/home/zjj/xjd/SAAS/")
import matplotlib.pyplot
import numpy as np
import scipy
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import math
from torch.utils.data import DataLoader
import time
from torchvision.utils import save_image

from configs.config import config


class MyDataset(data.Dataset):
    def __init__(self, clean_root, haze_root, transform=None):
        """img_folders = os.listdir(root)
        # print(img_folders)
        self.imgs = []
        for img_folder in img_folders:
            images = os.listdir(root+'/'+img_folder)
            # print(imgs)
            self.imgs = self.imgs + ([os.path.join(root+'/'+img_folder, img) for img in images])
        """
        self.data = []
        self.label = []
        resi = [
            (clean_root + "residential/" + file, haze_root + "residential/" + file)
            for file in os.listdir(clean_root + "residential")
        ]
        bk = [
            (clean_root + "back/" + file, haze_root + "back/" + file)
            for file in os.listdir(clean_root + "back")
        ]
        for _ in range(len(resi)):
            self.label.append(1)
        for _ in range(len(bk)):
            self.label.append(0)
        self.data = resi + bk
        self.transforms = transform
        # print(self.label)
        # print(self.imgs)

    def __getitem__(self, index):
        data_path = self.data[index]
        # label = math.floor(index / 100)
        # print(img_path.split('/')[-2])
        label = np.array(self.label[index])
        #         label = torch.from_numpy(label)
        data = (Image.open(data_path[0]), Image.open(data_path[1]))
        #         data = torch.from_numpy(data)

        if self.transforms:
            data = (self.transforms(data[0]), self.transforms(data[1]))
            label = torch.from_numpy(label)
        # save_image(data, './save/'+str(index)+'.png')  # 为0返回图片数据, save_img能将Tensor保存成图片
        return {
            "image": data[0],
            "img_haz": data[1],
            "label": label,
            "fname": os.path.basename(data_path),
        }

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = MyDataset(config.train_dir, config.train_haze_dir, data_transform)
    dataloder = DataLoader(dataset, batch_size=2, shuffle=False)
    print(dataloder)
    for batch, [data, label] in enumerate(dataloder):
        print(data[0].size())
        # print(label)
        # if data[0].shape == (2, 2, 610):
        #     print(data)  # 输出为空，shape都为(2, 2, 610)
