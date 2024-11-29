from collections import OrderedDict
import sys, os
sys.path.append('/home/zjj/xjd/SAAS/')
# print(sys.path)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import normalize, resize, to_pil_image
from dataloader import MyDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.models as models
import copy

from utils import args

if args.data_name == "geo":
    from configs.config import config
else:
    from configs.configSPOT import config

from models.ResNetlw import resnet_instance
from models.loser import CAMRefineLoss
from PIL import Image

assert args.cam == 'cam'

MODE = args.mode


def CAM_Loss(cln_cam, haz_cam):
#     print(cln_cam.size())
#     print(haz_cam.size())
    # print(cln_cam.max(dim=0))
    sc = cln_cam.size()[0]*cln_cam.size()[1]*cln_cam.size()[2]*cln_cam.size()[3]
    loss = (torch.sum((cln_cam-haz_cam)**2))/cln_cam.size()[0]

    return torch.sqrt(loss+1e-8)
    
def ClsConsis_Loss(cln_fc, haz_fc):
#     print(cln_fc)
#     print(haz_fc)
    # cln_fc, haz_fc = F.softmax(cln_fc, dim=1), F.softmax(haz_fc, dim=1)
    sc = cln_fc.size()[0]*cln_fc.size()[1]
    loss = (torch.sum((cln_fc-haz_fc)**2))/sc
    # print(loss)
    return torch.sqrt(loss+1e-8)/10

def train_haze():
    train_dataset = MyDataset(config.CLEAR.train_dir, MODE.train_dir,transform=args.T)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batchsize1,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    val_dataset = MyDataset(config.CLEAR.val_dir, MODE.val_dir,transform=args.T)
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=args.batchsize1,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    # test_dataset = datasets.ImageFolder(root=config.test_dir,transform=data_transform)
    # test_dataloader = DataLoader(dataset=test_dataset,
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               num_workers=4)

    print('train_dataset: {}'.format(len(train_dataset)))
    print('val_dataset: {}'.format(len(val_dataset)))
    # print('test_dataset: {}'.format(len(test_dataset)))


    device = torch.device(args.device)

    # 载入目标网络 
    model_name = f'{args.cls_model}_{MODE.name}'
    if args.cls_model == "resnet":
        tmodel = resnet_instance(n_class=args.num_classes, pretrained=False)
        tmodel.load_state_dict(torch.load(config.CLEAR.model_path, map_location=device))
    # print(tmodel)
    tmodel.to(device)

    #载入源网络
    smodel = resnet_instance(n_class=args.num_classes, pretrained=False)
    smodel.load_state_dict(torch.load(config.CLEAR.model_path, map_location=device))
    smodel.to(device)
    smodel.eval()
    

    # 优化方法、损失函数
    optimizer = optim.Adam(tmodel.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = [nn.CrossEntropyLoss(), CAM_Loss, ClsConsis_Loss, CAMRefineLoss()]
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    ## 训练
    num_epoch = config.num_epoch
    # 训练日志保存
    logfile_dir = config.logfile_dir

    acc_best_wts = tmodel.state_dict()
    best_train_acc = 0
    best_acc = 0
    iter_count = 0
    a = 0.001
    b = 0.1
    alpha = 0.01
    for epoch in range(num_epoch):
        train_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total = 0

        val_loss = 0.0
        val_acc = 0.0
        val_correct = 0
        val_total = 0

        for i, [data, labels] in enumerate(train_dataloader):
            # print(sample_batch)
            inputs_cln = data[0].to(device)
            inputs_haz = data[1].to(device)
            labels = labels.to(device)
            # print(labels)
            

            # 模型设置为train
            tmodel.train()

            # forward
            cams_cln, outputs_cln = smodel(inputs_cln)
            cams_haz, outputs_haz = tmodel(inputs_haz)
            
            # print(outputs.size())
            # loss
            # print(f'cross_entropy:{criterion[0](outputs_haz, labels)}, cam:{criterion[1](cams_cln, cams_haz)}, cls:{criterion[2](outputs_cln, outputs_haz)}, camref:{criterion[3](cams_cln, cams_haz, inputs_haz)}')
            loss = criterion[0](outputs_haz, labels) + a*criterion[1](cams_cln, cams_haz) + b*criterion[2](outputs_cln, outputs_haz) + alpha*criterion[3](cams_cln, cams_haz, inputs_haz)

            # forward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_correct += (torch.max(outputs_haz, 1)[1] == labels).sum().item()
            train_total += labels.size(0)

            # print('iter:{}'.format(i))
            
            if i % 10 == 9:
                for j, [data, labels] in enumerate(val_dataloader):
                    inputs_cln = data[0].to(device)
                    inputs_haz = data[0].to(device)
                    labels = labels.to(device)

                    tmodel.eval()
                    with torch.no_grad():
                        cams_cln, outputs_cln = smodel(inputs_cln)
                        cams_haz, outputs_haz = tmodel(inputs_haz)

                        # print(f'cross_entropy:{criterion[0](outputs_haz, labels)}, cam:{criterion[1](cams_cln, cams_haz)}, cls:{criterion[2](outputs_cln, outputs_haz)}, camref:{criterion[3](cams_cln, cams_haz, inputs_haz)}')
                        loss = criterion[0](outputs_haz, labels) + a*criterion[1](cams_cln, cams_haz) + b*criterion[2](outputs_cln, outputs_haz) + alpha*criterion[3](cams_cln, cams_haz, inputs_haz)
                    _, prediction = torch.max(outputs_haz, 1)
                    val_correct += ((labels == prediction).sum()).item()
                    val_total += inputs_cln.size(0)
                    val_loss += loss.item()
                
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                print('[{},{}] train_loss = {:.5f} train_acc = {:.5f} val_loss = {:.5f} val_acc = {:.5f}'.format(
                    epoch + 1, i + 1, train_loss / len(train_dataloader),train_correct / train_total, val_loss/len(val_dataloader),
                    val_correct / val_total))
                if val_acc >= best_acc:
                    if val_acc > best_acc or (val_acc==best_acc and train_acc>best_train_acc):
                        best_train_acc = train_acc
                        best_acc = val_acc
                        acc_best_wts = copy.deepcopy(tmodel.state_dict())

                with open(logfile_dir +'train_loss.txt','a') as f:
                    f.write(str(train_loss / len(train_dataloader)) + '\n')
                with open(logfile_dir +'train_acc.txt','a') as f:
                    f.write(str(train_correct / train_total) + '\n')
                with open(logfile_dir +'val_loss.txt','a') as f:
                    f.write(str(val_loss/len(val_dataloader)) + '\n')
                with open(logfile_dir +'val_acc.txt','a') as f:
                    f.write(str(val_correct / val_total) + '\n')

                iter_count += 200
                
                train_loss = 0.0
                train_total = 0
                train_correct = 0
                val_correct = 0
                val_total = 0
                val_loss = 0
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    alpha *= 1.2


    print('Train finish!')
    # 保存模型
    model_path = MODE.model_path
    # with open(model_file+'/model_squeezenet_teeth_1.pth','a') as f:
    #     torch.save(acc_best_wts,f)
    torch.save(acc_best_wts, model_path)
    print('Model save ok!')


#测试
def test_haze():
    model_name = f'{args.cls_model}_{MODE.name}'
    model = resnet_instance(n_class=args.num_classes, pretrained=False)
    #     print(model)
    # model.to(device)
    model.load_state_dict(torch.load(MODE.model_path, map_location=args.device))
    # print(model)
    model.eval()

    correct = 0
    total = 0
    acc = 0.0
    for file in os.listdir(MODE.test_dir):
        if 'checkpoints' in file:
            continue
        img = Image.open(f'{MODE.test_dir}{file}')
        inputs = normalize(torch.tensor(np.array(resize(img, (256, 256))).transpose(2,0,1) / 255.), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        _, outputs = model(inputs.unsqueeze(0).to(torch.float32))
        print(outputs)
        _, prediction = torch.max(outputs, 1)
        print(prediction)
    #     correct += (labels == prediction).sum().item()
    #     total += labels.size(0)

    # acc = correct / total
    # print('test finish, total:{}, correct:{}, acc:{:.3f}'.format(total, correct, acc))

if __name__=="__main__":
    train_haze()
    test_haze()
