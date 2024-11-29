import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image
import sys 
sys.path.append('/home/zjj/xjd/SAAS/train/segmentation/') 
from train.segmentation.fsan import Feedback_saliency_generator
from train.segmentation.geoconfig import config, args
if args.data_name == "geo":
    from configs.config import config as cg
else:
    from configs.configSPOT import config as cg


def segmentation_result():
    file = '3_3_45.tif'
    # Get your smap
    # cmap =cm.get_cmap("seismic")
    # cmap =cm.get_cmap("jet")
    cmap =cm.get_cmap("binary")
    smap = Image.open(os.path.join(f'train/segmentation/pred/{cg.NAME}/val_pred/', file))
    # img = Image.open(os.path.join('../data/geoeye-1/geo_test/img/', '3_6_19.tif'))
    img = np.asarray(Image.open(os.path.join(cg.CLEAR.test_dir, file)).resize((256, 256), Image.Resampling.LANCZOS))
    # img = np.asarray(Image.open(os.path.join('../data/spot5/spot5_test/img/', '2_13_20.tif')).resize((256, 256), Image.Resampling.LANCZOS))
    # smap = Image.open(os.path.join('./dataset/geoeye-1/train_deephaze_psm/', '3_7_18.tif'))
    # smap = Image.open(os.path.join('../data/geoeye-1/deep_haze/train/residential/', '3_8_18.tif'))
    smap = np.asarray(smap)
    print(smap.min(),smap.max())
    # smap = (255*(smap-smap.min())/(smap.max()-smap.min())).astype(np.uint8)
    _, smap= cv2.threshold(255-np.asarray(smap), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(np.asarray(smap))
    # sha = smap.shape
    # smap = 255-smap+np.random.randn(sha[0],sha[1])*10
    # smap[smap>80]=255
    # smap[smap<=80]=0
    # smap += np.random.randn(sha[0],sha[1])*0.8
    smap = (255*(cmap(np.asarray(255-smap)))[:,:,:3])
    smap = (smap+img)
    smap[smap>255]=0
    smap = smap.astype(np.uint8)

    # print(smap)
    # ssmap = (255 * cmap(np.asarray(smap) ** 2)[:, :, :3]).astype(np.uint8)


    cv2.imwrite(f'visual/segmentation/{file}', smap)


def fsan_segmentation_result(file):
    filename = file + '.tif'
    
    D_num = 1
    T_num = 3
    refine = 1
    feature_num = config.TRAIN.feature_num
    model_dir = config.TRAIN.model_dir
    image_size = config.TRAIN.image_size
    device = config.device
    FBSN = Feedback_saliency_generator(features = feature_num, D = D_num, T_int = T_num,if_refine = refine).to(device)
    def load_model(model, path):
        #  restore models
        model.load_state_dict(torch.load(path+'best weights.pth'))
    
    load_model(FBSN, model_dir)
    FBSN.to(device)
    FBSN.eval()
    def toTensor(img):
        assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        #return img.float().div(255).unsqueeze(0)  # 255也可以改为25
        return img.unsqueeze(0)

    val_input = np.array(cv2.resize(cv2.imread(os.path.join(config.TRAIN.test_dir, filename)), (image_size, image_size)), np.float32) / 255.
    test_x_tensor = toTensor(val_input)
    test_x_tensor = test_x_tensor.to(device)
    output = FBSN(test_x_tensor)

    smap = output[-1].cpu() #[1, 2, 256, 256]
    smap = smap.detach().numpy()  # (1, 2, 256, 256)
    smap = np.array(smap * 255., dtype='uint8')
    smap = np.squeeze(smap)[0, :, :] #(256,256)

    # Get your smap
    # cmap =cm.get_cmap("seismic")
    # cmap =cm.get_cmap("jet")
    cmap =cm.get_cmap("binary")
    # smap = Image.open(os.path.join(f'train/segmentation/pred/{config.NAME}/val_pred/', file))
    # img = Image.open(os.path.join('../data/geoeye-1/geo_test/img/', '3_6_19.tif'))
    img = np.asarray(Image.open(os.path.join(cg.CLEAR.test_dir, filename)).resize((256, 256), Image.Resampling.LANCZOS))
    # img = np.asarray(Image.open(os.path.join('../data/spot5/spot5_test/img/', '2_13_20.tif')).resize((256, 256), Image.Resampling.LANCZOS))
    # smap = Image.open(os.path.join('./dataset/geoeye-1/train_deephaze_psm/', '3_7_18.tif'))
    # smap = Image.open(os.path.join('../data/geoeye-1/deep_haze/train/residential/', '3_8_18.tif'))
    # smap = np.asarray(smap)
    print(smap.min(),smap.max())
    # smap = (255*(smap-smap.min())/(smap.max()-smap.min())).astype(np.uint8)
    _, smap= cv2.threshold(255-np.asarray(smap), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(np.asarray(smap))
    # sha = smap.shape
    # smap = 255-smap+np.random.randn(sha[0],sha[1])*10
    # smap[smap>80]=255
    # smap[smap<=80]=0
    # smap += np.random.randn(sha[0],sha[1])*0.8
    smap = (255*(cmap(np.asarray(255-smap)))[:,:,:3])
    smap = (smap+img)
    smap[smap>255]=0
    smap = smap.astype(np.uint8)

    # print(smap)
    # ssmap = (255 * cmap(np.asarray(smap) ** 2)[:, :, :3]).astype(np.uint8)

    if not os.path.exists('visual/segmentation/'):
        os.makedirs('visual/segmentation/')
    a = cv2.imwrite(f"/home/zjj/xjd/SAAS/visual/segmentation/{file}_{args.mode.name}_{args.cam}_{args.dehaze}_{args.net}_{args.wo}.png", smap)
    print(f'保存:{a}')

if __name__=="__main__":
    # segmentation_result()
    fsan_segmentation_result('3_6_19')
