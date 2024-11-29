import select
from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import numpy as np
from models.layer_factory import CRPBlock

np.random.seed(47)

__all__ = {18:'resnet18', 34:'resnet34', 50:'resnet50', 101:'resnet101',
           152:'resnet152'}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )

class PALayer(nn.Module):
    def __init__(self, channel, kernel_size=1, dilation=1):
        super(PALayer, self).__init__()
        assert kernel_size in [1, 3, 5], 'kernel size must be 1 or 3 or 5'
        _kernel_size = kernel_size+(kernel_size-1)*(dilation-1)
        padding = int((_kernel_size-1)/2)
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, kernel_size, padding=padding, dilation=dilation, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, kernel_size, padding=padding, dilation=dilation, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CSALayer(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(CSALayer, self).__init__()
         
        self.ca = CALayer(in_channel)
        kernel_size = [1, 3, 5]
        dilation = [1, 2, 2]
        self.pa_modules=nn.ModuleList([PALayer(in_channel, kernel_size[i], dilation[i]) for i in range(3)])
        self.conv1 = nn.Conv2d(in_channel, out_channels, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channel, out_channels, 1, padding=0, bias=True)
 
    def forward(self, x):
        x1 = self.ca(x)
        xlst = [self.pa_modules[i](x1) for i in range(3)]
        res = sum(xlst)
        res = self.conv1(res) + self.conv2(x)

        return res

class ShuffleAttentionModule(nn.Module):
    def __init__(self, in_channel):
        super(ShuffleAttentionModule, self).__init__()

        self.factor = 8
        _in_channel = in_channel // self.factor
        self.ha = nn.Sequential(
            nn.Conv2d(_in_channel, _in_channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_in_channel // 8, _in_channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        in_channel = np.array([i for i in range(x.size(1))])
        np.random.shuffle(in_channel)
        x_shuffle = x[:,torch.from_numpy(in_channel),:,:]
        out = []
        length = x.size(1) // self.factor
        for i in range(self.factor):
            out.append(self.ha(x_shuffle[:,i*length:(i+1)*length]))

        return torch.cat(out, dim=1) * x
        # return self.ha(x) * x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualAttentionBlock, self).__init__()
        
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
        self.sam = ShuffleAttentionModule(in_channel)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.sam(out)

        return out + x

# # 搭建CA_ResNet34
# class BottleneckBlock(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
#                  norm_layer=None):
#         super(BottleneckBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
#         self.bn1 = norm_layer(width)
#         self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, stride=stride, groups=groups, dilation=dilation,
#                                bias=False)
#         self.bn2 = norm_layer(width)
#         self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU()
#         self.downsample = downsample
#         self.stride = stride
#         # self.ca = CoordAttention(in_channels=planes * self.expansion, out_channels=planes * self.expansion)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         # out = self.ca(out)  # add CA
#         out += identity
#         out = self.relu(out)

#         return out


# "Multi-resolution attention collaboration network"
# class ResNet(nn.Module):

#     def __init__(self, block, depth, n_class=1000, with_pool=True):
#         super(ResNet, self).__init__()
#         layer_cfg = {
#             18: [2, 2, 2, 2],
#             34: [3, 4, 6, 3],
#             50: [3, 4, 6, 3],
#             101: [3, 4, 23, 3],
#             152: [3, 8, 36, 3]
#         }
#         layers = layer_cfg[depth]
#         self.num_classes = n_class
#         self.with_pool = with_pool
#         self._norm_layer = nn.BatchNorm2d

#         self.inplanes = 64
#         self.dilation = 1

#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = self._norm_layer(self.inplanes)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         self.p_l4 = conv1x1(2048, 512, bias=False)
#         self.rsb_l4 = ResidualAttentionBlock(512)
#         self.b_l4 = conv1x1(512, 256, bias=False)

#         self.p_l3 = conv1x1(1024, 256, bias=False)
#         self.rsb_l3 = ResidualAttentionBlock(256)
#         self.b_l3 = conv1x1(256, 128, bias=False)

#         self.p_l2 = conv1x1(512, 128, bias=False)
#         self.rsb_l2 = ResidualAttentionBlock(128)
#         self.b_l2 = conv1x1(128, 128, bias=False)

#         self.p_l1 = conv1x1(256, 128, bias=False)
#         self.rsb_l1 = ResidualAttentionBlock(128)
#         self.b_l1 = conv1x1(128, 128, bias=False)

#         self.clf_conv = nn.Conv2d(
#             128, n_class, kernel_size=3, stride=1, padding=1, bias=True
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
#                 norm_layer(planes * block.expansion), )

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         out1 = x
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x1 = self.layer1(x)
#         out2 = x1
#         x2 = self.layer2(x1)
#         out3 = x2
#         x3 = self.layer3(x2)
#         out4 = x3
#         x4 = self.layer4(x3)
#         out5 = x4

#         l4 = self.p_l4(x4)
#         l4 = self.relu(l4)
#         l4 = self.rsb_l4(l4)
#         l4 = self.b_l4(l4)
#         l4 = nn.Upsample(size=x3.size()[2:], mode='bilinear', align_corners=True)(l4)

#         l3 = self.p_l3(x3)
#         l3 = l3 + l4
#         l3 = self.relu(l3)
#         l3 = self.rsb_l3(l3)
#         l3 = self.b_l3(l3)
#         l3 = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)(l3)

#         l2 = self.p_l2(x2)
#         l2 = l2 + l3
#         l2 = self.relu(l2)
#         l2 = self.rsb_l2(l2)
#         l2 = self.b_l2(l2)
#         l2 = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(l2)

#         l1 = self.p_l1(x1)
#         l1 = l1 + l2
#         l1 = self.relu(l1)
#         l1 = self.rsb_l1(l1)

#         out = self.clf_conv(l1)
#         out6 = nn.Sigmoid()(out)

#         return out1, out2, out3, out4, out5, out6

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, depth, num_classes=21):
        self.inplanes = 64
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
   
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(
            256, num_classes, kernel_size=3, stride=1, padding=1, bias=True
        )

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        out1 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        out2 = l1
        l2 = self.layer2(l1)
        out3 = l2
        l3 = self.layer3(l2)
        out4 = l3
        l4 = self.layer4(l3)
        out5 = l4
        
        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        out6 = nn.Sigmoid()(out)

        return out1, out2, out3, out4, out5, out6

    def get_1x_lr_params_NOscale(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)


        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        b.append(self.clf_conv.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i



    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]

def ca_resnet50(**kwargs):
    return ResNet(BottleneckBlock, 50, **kwargs)


def resnet_instance(n_class, depth=50, pretrained=False, **kwargs):  # resnet50的模型
    model = ResNet(BottleneckBlock, depth, n_class, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls[__all__[50]])
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, n_class)  # 15 output classes
    # stdv = 1.0 / math.sqrt(1000)
    # for p in model.fc.parameters():
    #     p.data.uniform_(-stdv, stdv)

    return model

if __name__ == "__main__":
    # 利用高阶 API 查看模型
    ca_res50 = resnet_instance(n_class=2, pretrained=True, with_pool=False).cuda()
    print(ca_res50)
    x = torch.rand(1, 3, 224, 224).cuda()
    i = ca_res50(x)
    print(i.shape)
    summary(ca_res50, (3, 224, 224))
