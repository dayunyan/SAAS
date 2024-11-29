import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import numpy as np

__all__ = {18:'resnet18', 34:'resnet34', 50:'resnet50', 101:'resnet101',
           152:'resnet152'}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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

# 搭建CA_ResNet34
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, stride=stride, groups=groups, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        # self.ca = CoordAttention(in_channels=planes * self.expansion, out_channels=planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = self.ca(out)  # add CA
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, depth, n_class=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = n_class
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.bn = self._norm_layer(512 * block.expansion)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # # self.conv_1 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, stride=1, padding=1)
        # # self.conv_2 = nn.Conv2d(128 * block.expansion, 128 * block.expansion, kernel_size=3, stride=1, padding=1)
        # # self.conv_3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, stride=1, padding=1)
        # # self.conv_4 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1)
        # # self.csa=CSALayer(block.expansion * (64+128+256+512), 512 * block.expansion)
        # self.csa1=CSALayer(64 * block.expansion, 512 * block.expansion)
        # self.csa2=CSALayer(128 * block.expansion, 512 * block.expansion)
        # self.csa3=CSALayer(256 * block.expansion, 512 * block.expansion)
        # self.csa=CSALayer(512 * block.expansion, 512 * block.expansion)
        # self.post_conv1 = nn.Conv2d(64, 512 * block.expansion, kernel_size=1, stride=8, padding=0, bias=False)
        # self.post_conv = nn.Sequential(nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1),
        #                                 nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1))
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if n_class > 0:
            self.fc = nn.Linear(512 * block.expansion, n_class)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)
        # layers=[x1,x2,x3]
        # downsample=[F.interpolate(y, size=[x4.size()[-2], x4.size()[-1]], mode='bilinear') for y in layers]
        # x4 = x4 + self.csa1(downsample[0]) + self.csa2(downsample[1]) + self.csa3(downsample[2])
        # print(x1.size(),x2.size(),x3.size(),x4.size())
        # x1, x2, x3, x4 = self.conv_1(x1), self.conv_2(x2), self.conv_3(x3), self.conv_4(x4)
        # layers=[x2,x3,x4]
        # downsample=[F.interpolate(y, size=[x1.size()[-2], x1.size()[-1]], mode='bilinear') for y in layers]
        # catx = torch.cat((x1, downsample[0], downsample[1], downsample[2]), 1)
        # x = self.csa(x4)
        # x = self.bn(x)
        # x = self.post_conv(catx)
        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = self.dropout(x)
        # x = self.softmax(x)

        return x

def ca_resnet50(**kwargs):
    return ResNet(BottleneckBlock, 50, **kwargs)


def resnet_PCA_instance(n_class, depth=50, pretrained=False, **kwargs):  # resnet50的模型
    model = ResNet(BottleneckBlock, depth, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls[__all__[50]])
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)  # 15 output classes
    stdv = 1.0 / math.sqrt(1000)
    for p in model.fc.parameters():
        p.data.uniform_(-stdv, stdv)

    return model

if __name__ == "__main__":
    # 利用高阶 API 查看模型
    ca_res50 = resnet_PCA_instance(n_class=2, pretrained=True, with_pool=False).cuda()
    print(ca_res50)
    x = torch.rand(1, 3, 224, 224).cuda()
    i = ca_res50(x)
    print(i.shape)
    summary(ca_res50, (3, 224, 224))
