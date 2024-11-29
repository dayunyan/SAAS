import os, sys
sys.path.append('.')
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import normalize, resize, to_pil_image
from dataloader import MyDataset
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.models import vgg16
import copy
from collections import OrderedDict
import torch
from torchcam.methods import XGradCAM

# from config import config
from configSPOT import config

from PerceptualLoss import LossNetwork as PerLoss
from AttentionResNet import resnet_PCA_instance
from PIL import Image


batch_size = config.batch_size
num_workers = config.num_workers
num_classes = config.num_classes
MODE = config.DEEPHAZE #config.HAZE
 
data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """
    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}	# 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers) # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            if name == 'fc':
                x = x.view(-1,2048)
#             print(x.size())
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def Semantic_Loss(cln_sem, haz_sem):
#     print(cln_sem.size())
#     print(haz_sem.size())
    sc = cln_sem.size()[0]*cln_sem.size()[1]*cln_sem.size()[2]*cln_sem.size()[3]
    loss = (torch.sum((cln_sem-haz_sem)**2))/sc
    # print(loss)
    return loss

def CAM_Loss(cln_cam, haz_cam):
#     print(cln_cam.size())
#     print(haz_cam.size())
    print(cln_cam.max(dim=0))
    sc = cln_cam.size()[0]*cln_cam.size()[1]*cln_cam.size()[2]*cln_cam.size()[3]
    loss = (torch.sum((cln_cam/255.-haz_cam/255.)**2))/cln_cam.size()[0]
    # print(loss)
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
    train_dataset = MyDataset(config.CLEAR.train_dir, MODE.train_dir,transform=data_transform)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    val_dataset = MyDataset(config.CLEAR.val_dir, MODE.val_dir,transform=data_transform)
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    # test_dataset = datasets.ImageFolder(root=config.test_dir,transform=data_transform)
    # test_dataloader = DataLoader(dataset=test_dataset,
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               num_workers=4)

    print('train_dataset: {}'.format(len(train_dataset)))
    print('val_dataset: {}'.format(len(val_dataset)))
    # print('test_dataset: {}'.format(len(test_dataset)))


    # 载入目标网络 
    model_name = f'pcares_{MODE.name}'
    tmodel = resnet_PCA_instance(n_class=num_classes, pretrained=True)
    tmodel.to(device)

    #载入源网络
    smodel = resnet_PCA_instance(n_class=num_classes, pretrained=False)
    smodel.load_state_dict(torch.load(config.CLEAR.pcares_path, map_location=device))
    smodel.to(device)
    smodel.eval()

    #加钩子
    global sfeatures_out_hook, tfeatures_out_hook
    def shook(module, fea_in, fea_out):
        sfeatures_out_hook.append(fea_out)
        return None
    
    def thook(module, fea_in, fea_out):
        tfeatures_out_hook.append(fea_out)
        return None
    
    return_layers = {'csa':'csa', 'fc':'fc'}
    for (name, module) in smodel.named_modules():
        if name == return_layers['csa']:
            module.register_forward_hook(hook=shook)
        elif name == return_layers['fc']:
            module.register_forward_hook(hook=shook)
        else:
            pass

    for (name, module) in tmodel.named_modules():
        if name == return_layers['csa']:
            module.register_forward_hook(hook=thook)
        elif name == return_layers['fc']:
            module.register_forward_hook(hook=thook)
        else:
            pass
    
    # Set CAM extractor
    tcam_extractor = XGradCAM(tmodel,target_layer='csa')
    scam_extractor = XGradCAM(smodel,target_layer='csa')
    print(scam_extractor)
    cmap = cm.get_cmap("jet")
    
    # 优化方法、损失函数
    optimizer = optim.Adam(tmodel.parameters(),lr=0.001,weight_decay=1e-5)
    criterion = []
    criterion.append(nn.CrossEntropyLoss())
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))
#     loss_fc = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,10, 0.1)

    ## 训练
    num_epoch = config.num_epoch
    # 训练日志保存
    logfile_dir = config.logfile_dir

    acc_best_wts = tmodel.state_dict()
    best_train_acc = 0
    best_acc = 0
    iter_count = 0
    
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
#             print(inputs_cln.size())
            
#             tmodel = copy.deepcopy(tmodel_final)
        
            global sfeatures_out_hook
            global tfeatures_out_hook
            
            sfeatures_out_hook=[]
            tfeatures_out_hook=[]

            sfeatures_out_hook.append(smodel(inputs_cln))
            tfeatures_out_hook.append(tmodel(inputs_haz))
            # tmodel(inputs_haz)

            # print(features_in_hook)  # 勾的是指定层的输入
            # print(sfeatures_out_hook)  # 勾的是指定层的输出
            # print(tfeatures_out_hook)

            # 模型设置为train
            tmodel.train()
#             tmodel_final.train()

            # forward
            out_cln_sem, out_cln_fc, out_cln_hat = sfeatures_out_hook
            out_haz_sem, out_haz_fc, out_haz_hat = tfeatures_out_hook
            out_cln = out_cln_hat
            out_haz = out_haz_hat
            # cam
            batch = inputs_cln.size()[0]
            out_cln_cam = np.zeros((batch, 1,  config.weight, config.height))
            out_haz_cam = np.zeros((batch, 1,  config.weight, config.height))
#             sactivation_map = scam_extractor(out_cln.argmax(dim=1).cpu().numpy().tolist(), out_cln)
#             tactivation_map = tcam_extractor(out_haz.argmax(dim=1).cpu().numpy().tolist(), out_haz)
            sactivation_map = scam_extractor(out_cln.argmax(dim=1).cpu().numpy().tolist(), out_cln, retain_graph=True)
            tactivation_map = tcam_extractor(out_haz.argmax(dim=1).cpu().numpy().tolist(), out_haz, retain_graph=True)
            for j in range(batch):
#                 print('----train_batch:{}'.format(j))
#                 print(out_cln[i, :].argmax().item(), out_cln[i, :].unsqueeze(0))
#                 print(out_cln.size())
#                 print(len(sactivation_map))
#                 print(sactivation_map[0].size())
                sact_pil_map = to_pil_image(sactivation_map[0][j,:,:], mode='F')
#                 print(np.array(sact_pil_map))
                ssmap = sact_pil_map.resize((config.weight, config.height), Image.Resampling.LANCZOS)
#                 ssmap = (255 * cmap(np.asarray(ssmap) ** 2)[:, :, :3]).astype(np.uint8)
                out_cln_cam[j,:,:,:] = np.array(ssmap)
                
                tact_pil_map = to_pil_image(tactivation_map[0][j,:,:], mode='F')
    #           print(np.array(act_pil_map))
                tsmap = tact_pil_map.resize((config.weight, config.height), Image.Resampling.LANCZOS)
                out_haz_cam[j,:,:,:] = np.array(tsmap)
            out_cln_cam = torch.from_numpy(out_cln_cam)*255.
            out_haz_cam = torch.from_numpy(out_haz_cam)*255.
            # print(out_cln_cam)
            # loss
#             a = b = 1
#             loss = loss_fc(out_haz, labels)+a*Semantic_Loss(out_cln_sem,out_haz_sem)+b*CAM_Loss(out_cln_cam,out_haz_cam)+ClsConsis_Loss(out_cln_fc,out_haz_fc)
            a = 0.1
            b = 0.01
            out_cln_cam = torch.cat([out_cln_cam, out_cln_cam+a*torch.randn((batch,1,config.weight,config.height)), out_cln_cam+a*torch.randn((batch,1,config.weight,config.height))], 1).type(torch.float).cuda()
            out_haz_cam = torch.cat([out_haz_cam, out_haz_cam+a*torch.randn((batch,1,config.weight,config.height)), out_haz_cam+a*torch.randn((batch,1,config.weight,config.height))], 1).type(torch.float).cuda()
            print(f'train:{ClsConsis_Loss(out_cln_fc,out_haz_fc)},{criterion[0](out_haz,labels)},{criterion[1](out_cln_cam,out_haz_cam)}')
            loss=ClsConsis_Loss(out_cln_fc,out_haz_fc) + criterion[0](out_haz,labels) + b*criterion[1](out_cln_cam,out_haz_cam).cpu()

            # forward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#             for name, param in tmodel.named_parameters():
#                 param.requires_grad = True

            # 统计
            train_loss += loss.item()
            train_correct += (torch.max(out_haz_hat, 1)[1] == labels).sum().item()
            train_total += labels.size(0)

            print('iter:{}'.format(i))

            if i % 10 == 9:
                tmodel.eval()
#                 tmodel = copy.deepcopy(tmodel_final)
                
                for j, [data, labels] in enumerate(val_dataloader):
                    inputs_cln = data[0].to(device)
                    inputs_haz = data[0].to(device)
                    labels = labels.to(device)

                    sfeatures_out_hook = []
                    tfeatures_out_hook = []
                    sfeatures_out_hook.append(smodel(inputs_cln))
                    tfeatures_out_hook.append(tmodel(inputs_haz))
                    
                    # forward
                    out_cln_sem, out_cln_fc, out_cln = sfeatures_out_hook
                    out_haz_sem, out_haz_fc, out_haz = tfeatures_out_hook

                    # cam
                    batch = inputs_cln.size()[0]
                    out_cln_cam = np.zeros((batch, 1,  config.weight, config.height))
                    out_haz_cam = np.zeros((batch, 1,  config.weight, config.height))
                    sactivation_map = scam_extractor(out_cln.argmax(dim=1).cpu().numpy().tolist(), out_cln)
                    tactivation_map = tcam_extractor(out_haz.argmax(dim=1).cpu().numpy().tolist(), out_haz)
                    for k in range(batch):
#                         print('----val_batch:{}'.format(j))
                        sact_pil_map = to_pil_image(sactivation_map[0][k,:,:], mode='F')
            #           print(np.array(act_pil_map))
                        ssmap = sact_pil_map.resize((config.weight, config.height), Image.Resampling.LANCZOS)
                        out_cln_cam[k,:,:,:] = np.array(ssmap)

                        
                        tact_pil_map = to_pil_image(tactivation_map[0][k,:,:], mode='F')
            #           print(np.array(act_pil_map))
                        tsmap = tact_pil_map.resize((config.weight, config.height), Image.Resampling.LANCZOS)
                        out_haz_cam[k,:,:,:] = np.array(tsmap)
                    out_cln_cam = torch.from_numpy(out_cln_cam)*255
                    out_haz_cam = torch.from_numpy(out_haz_cam)*255
                    

        #             print(outputs)
                    # loss
#                     loss = loss_fc(out_haz, labels)+CAM_Loss(out_cln_cam,out_haz_cam)+ClsConsis_Loss(out_cln_fc,out_haz_fc)
                    
                    out_cln_cam = torch.cat([out_cln_cam, out_cln_cam+a*torch.randn((batch,1,config.weight,config.height)), out_cln_cam+a*torch.randn((batch,1,config.weight,config.height))], 1).type(torch.float).cuda()
                    out_haz_cam = torch.cat([out_haz_cam, out_haz_cam+a*torch.randn((batch,1,config.weight,config.height)), out_haz_cam+a*torch.randn((batch,1,config.weight,config.height))], 1).type(torch.float).cuda()
                    print(f'val:{ClsConsis_Loss(out_cln_fc,out_haz_fc)},{criterion[0](out_haz,labels)},{criterion[1](out_cln_cam,out_haz_cam)}')
                    loss=ClsConsis_Loss(out_cln_fc,out_haz_fc) + criterion[0](out_haz,labels) + b*criterion[1](out_cln_cam,out_haz_cam).cpu()
                    _, prediction = torch.max(out_haz, 1)
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

                with open(logfile_dir +'train_loss_'+model_name+'.txt','a') as f:
                    f.write(str(train_loss / 100) + '\n')
                with open(logfile_dir +'train_acc_haze_'+model_name+'.txt','a') as f:
                    f.write(str(train_correct / train_total) + '\n')
                with open(logfile_dir +'val_loss_haze_'+model_name+'.txt','a') as f:
                    f.write(str(val_loss/len(val_dataloader)) + '\n')
                with open(logfile_dir +'val_acc_haze_'+model_name+'.txt','a') as f:
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


    print('Train finish!')
    # 保存模型
    model_file = config.model_file
    # with open(model_file+'/model_squeezenet_teeth_1.pth','a') as f:
    #     torch.save(acc_best_wts,f)
    torch.save(acc_best_wts, MODE.pcares_perloss_path)
    print('Model save ok!')

    

#测试
def test_haze():
    model = resnet_PCA_instance(n_class=num_classes, pretrained=True)
    model.load_state_dict(torch.load(MODE.pcares_perloss_path, map_location='cpu'))
    print(model)
    model.eval()

    correct = 0
    total = 0
    acc = 0.0
    for file in os.listdir(MODE.test_dir):
        if 'checkpoints' in file:
            continue
        img = Image.open(MODE.test_dir + file)
        inputs = normalize(torch.tensor(np.array(resize(img, (256, 256))).transpose(2,0,1) / 255.), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        outputs = model(inputs.unsqueeze(0).to(torch.float32))
        print(outputs)
        _, prediction = torch.max(outputs, 1)
        print(prediction)
#         correct += (labels == prediction).sum().item()
#         total += labels.size(0)

#     acc = correct / total
#     print('test finish, total:{}, correct:{}, acc:{:.3f}'.format(total, correct, acc))


if __name__=="__main__":
    train_haze()
    test_haze()