# Define your model
import os, sys


sys.path.append("/home/zjj/xjd/SAAS/")
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from models.ResNetlw import resnet_instance

from datasets.loveda import LoveDAClassification

from utils import args, config

if args.cam == "cam":
    from torchcam.methods import CAM as class_activation_map
elif args.cam == "gradcam":
    from torchcam.methods import GradCAM as class_activation_map
elif args.cam == "gradcam++":
    from torchcam.methods import GradCAMpp as class_activation_map
elif args.cam == "xgradcam":
    from torchcam.methods import XGradCAM as class_activation_map

MODE = args.mode
device = torch.device(args.device)


def make_cam(img_list, to_path, mode):
    if args.cls_model == "resnet":
        model = resnet_instance(n_class=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODE.model_path, map_location=device))
    model.to(device)
    model.eval()
    # print(to_path)

    cam_extractor = class_activation_map(model, "cls")

    # Get your input
    for file in tqdm(img_list, desc=f"{mode}:"):
        # print(file)
        if "checkpoints" in file:
            continue
        img = Image.open(file)
        width, hight = img.size
        #         print(np.array(img).shape)
        input_tensor = normalize(
            torch.tensor(np.array(resize(img, (256, 256))).transpose(2, 0, 1) / 255.0),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ).to(device)

        # Preprocess your data and feed it to the model
        cam, out = model(input_tensor.to(torch.float32).unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        # activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        act_pil_map = to_pil_image(cam[:, 1, :, :], mode="L")
        #         print(np.array(act_pil_map))
        smap = act_pil_map.resize((width, hight), Image.Resampling.LANCZOS)
        # print(np.array(smap))

        # save pseudo saliency map
        smap.save(os.path.join(to_path, os.path.basename(file)))
        # Visualize the raw CAM


#         plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

# Resize the CAM and overlay it
#         img_to_tensor = ToTensor()
#         img = img_to_tensor(img)
#         result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
#         plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()


def cat_full(org_dir, to_dir):
    to_files = os.listdir(to_dir)
    for file in os.listdir(org_dir):
        if "checkpoints" in file:
            continue
        img = Image.open(org_dir + file)
        if file in to_files:
            file = file[:-4] + "_099.tif"
        img.save(to_dir + file)
    print(len(os.listdir(org_dir)), len(to_files), len(os.listdir(to_dir)))


if __name__ == "__main__":
    train_dataset = LoveDAClassification(image_dir=MODE.train_dir, mask_dir=None)
    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     shuffle=False,
    # )

    val_dataset = LoveDAClassification(image_dir=MODE.val_dir, mask_dir=None)

    test_dataset = LoveDAClassification(image_dir=MODE.test_dir, mask_dir=None)

    if not os.path.exists(MODE.train_psm):
        os.makedirs(MODE.train_psm)
    if not os.path.exists(MODE.val_psm):
        os.makedirs(MODE.val_psm)
    if not os.path.exists(MODE.test_psm):
        os.makedirs(MODE.test_psm)

    make_cam(train_dataset.rgb_filepath_list, MODE.train_psm, "train")
    make_cam(val_dataset.rgb_filepath_list, MODE.val_psm, "val")
    make_cam(test_dataset.rgb_filepath_list, MODE.test_psm, "test")
