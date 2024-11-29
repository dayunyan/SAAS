# Define your model
import imp
import os, sys

sys.path.append("/home/zjj/xjd/SAAS/")
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from models.ResNetlw_woDARM import resnet_instance

from utils import args

if args.data_name == "geo":
    from configs.config import config
else:
    from configs.configSPOT import config

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


def make_cam(img_dir, to_path):
    if args.cls_model == "resnet":
        model = resnet_instance(n_class=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODE.model_woAtt_path, map_location=device))
    model.to(device)
    model.eval()
    # print(to_path)

    cam_extractor = class_activation_map(model, "cls")

    # Get your input
    for file in os.listdir(img_dir):
        # print(file)
        if "checkpoints" in file:
            continue
        img = Image.open(os.path.join(img_dir, file))
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
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        # get pseudo saliency map
        act_pil_map = to_pil_image(activation_map[0].squeeze(0), mode="L")
        #         print(np.array(act_pil_map))
        smap = act_pil_map.resize((width, hight), Image.Resampling.LANCZOS)
        # print(np.array(smap))

        # save pseudo saliency map
        smap.save(os.path.join(to_path, file))
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
    if not os.path.exists(MODE.train_woAtt_psm):
        os.makedirs(MODE.train_woAtt_psm)
    if not os.path.exists(MODE.val_woAtt_psm):
        os.makedirs(MODE.val_woAtt_psm)
    if not os.path.exists(MODE.test_woAtt_psm):
        os.makedirs(MODE.test_woAtt_psm)
    if not os.path.exists(MODE.aux_woAtt_psm):
        os.makedirs(MODE.aux_woAtt_psm)

    make_cam(os.path.join(MODE.train_dir, "residential"), MODE.train_woAtt_psm)
    make_cam(os.path.join(MODE.val_dir, "residential"), MODE.val_woAtt_psm)
    make_cam(MODE.test_dir, MODE.test_woAtt_psm)
    make_cam(MODE.aux_postive_dir, MODE.aux_woAtt_psm)

    if not os.path.exists(MODE.full_woAtt_train):
        os.mkdir(MODE.full_woAtt_train)
    else:
        shutil.rmtree(MODE.full_woAtt_train)
        os.mkdir(MODE.full_woAtt_train)

    if not os.path.exists(MODE.full_woAtt_train_gt):
        os.mkdir(MODE.full_woAtt_train_gt)
    else:
        shutil.rmtree(MODE.full_woAtt_train_gt)
        os.mkdir(MODE.full_woAtt_train_gt)

    cat_full(MODE.train_dir + "residential/", MODE.full_woAtt_train)
    cat_full(MODE.aux_postive_dir, MODE.full_woAtt_train)
    cat_full(MODE.train_woAtt_psm, MODE.full_woAtt_train_gt)
    cat_full(MODE.aux_woAtt_psm, MODE.full_woAtt_train_gt)
