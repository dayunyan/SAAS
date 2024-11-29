import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import numpy as np

__all__ = {
    18: "resnet18",
    34: "resnet34",
    50: "resnet50",
    101: "resnet101",
    152: "resnet152",
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class PALayer(nn.Module):
    def __init__(self, channel, kernel_size=1, dilation=1):
        super(PALayer, self).__init__()
        assert kernel_size in [1, 3, 5], "kernel size must be 1 or 3 or 5"
        _kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = int((_kernel_size - 1) / 2)
        self.pa = nn.Sequential(
            nn.Conv2d(
                channel,
                channel // 8,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel // 8,
                1,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.Sigmoid(),
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
            nn.Sigmoid(),
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
        self.pa_modules = nn.ModuleList(
            [PALayer(in_channel, kernel_size[i], dilation[i]) for i in range(3)]
        )
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        in_channel = np.array([i for i in range(x.size(1))])
        if self.training:
            np.random.shuffle(in_channel)
        x_shuffle = x[:, torch.from_numpy(in_channel), :, :]
        out = []
        length = x.size(1) // self.factor
        for i in range(self.factor):
            out.append(self.ha(x_shuffle[:, i * length : (i + 1) * length]))

        return torch.cat(out, dim=1) * x
        # return self.ha(x) * x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(ResidualAttentionBlock, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channels, 3, 1, 1, bias=False)
        self.sam = ShuffleAttentionModule(out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.sam(out)

        return out + self.conv(x)


# 搭建CA_ResNet34
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
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

    def __init__(self, block, depth, n_class=2, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        layers = layer_cfg[depth]
        self.layer_stride = [1, 1, 2, 2]
        self.num_classes = n_class
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.rab1 = ResidualAttentionBlock(64, 64 * 4)
        self.rab2 = ResidualAttentionBlock(256, 128 * 4)
        self.rab3 = ResidualAttentionBlock(512, 256 * 4)
        self.rab4 = ResidualAttentionBlock(1024, 512 * 4)
        self.cls = self.classifier(2048, n_class)

        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(512, out_planes, kernel_size=1, padding=0),  # fc8
            # nn.Sigmoid()
        )

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        # --------------------------
        atten_maps = torch.relu(atten_maps)
        batch_mins, _ = torch.min(
            atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True
        )
        batch_maxs, _ = torch.max(
            atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True
        )
        atten_normed = torch.div(
            atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
            batch_maxs - batch_mins + 1e-8,
        )
        # print(f'batch_maxs-batch_mins:{atten_normed}')
        atten_normed = atten_normed.view(atten_shape)

        # atten_normed = torch.sigmoid(atten_maps)

        return atten_normed

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                1,
                64,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x_1 = self.rab1(x0)
        x1 = nn.UpsamplingBilinear2d(size=(x1.size()[-2], x1.size()[-1]))(x_1) + x1

        x2 = self.layer2(x1)
        x_2 = self.rab2(x1)
        x2 = nn.UpsamplingBilinear2d(size=(x2.size()[-2], x2.size()[-1]))(x_2) + x2

        x3 = self.layer3(x2)
        x_3 = self.rab3(x2)
        x3 = nn.UpsamplingBilinear2d(size=(x3.size()[-2], x3.size()[-1]))(x_3) + x3

        x4 = self.layer4(x3)
        x_4 = self.rab4(x3)
        x4 = nn.UpsamplingBilinear2d(size=(x4.size()[-2], x4.size()[-1]))(x_4) + x4

        out = self.cls(x4)
        cam = self.normalize_atten_maps(out)

        if self.with_pool:
            out = self.avgpool(out)

        # out2 = self.softmax(out1)

        return cam, out.view(out.size(0), -1), [x1, x2, x3, x4]


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
    # print(ca_res50)
    x = torch.rand(1, 3, 512, 512).cuda()
    i = ca_res50(x)
    print(i.shape)
    summary(ca_res50, (3, 224, 224))
