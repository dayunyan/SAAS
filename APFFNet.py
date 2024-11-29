import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = default_conv(in_channels, growth_rate, kernel_size=kernel_size)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

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
    def __init__(self, channel, n):
        super(CALayer, self).__init__()
        self.n = n
        self.avg_pool1 = nn.AdaptiveAvgPool2d(n)
        # self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        # self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.ca = nn.Sequential(
                nn.Conv2d(channel*(self.n**2), channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool1(x).view(b, (self.n**2)*c)
        # y2 = self.avg_pool2(x).view(b, 4*c)
        # y3 = self.avg_pool4(x).view(b, 16*c)
        y = self.ca(y)
        return x * y

class CSALayer(nn.Module):
    def __init__(self, channel):
        super(CSALayer, self).__init__()
        n = [1, 2, 4]
        self.ca = nn.ModuleList([CALayer(channel, i) for i in n])
        kernel_size = [1, 3, 5]
        dilation = [1, 2, 2]
        self.pa_modules=nn.ModuleList([PALayer(channel, kernel_size[i], dilation[i]) for i in range(3)])
 
    def forward(self, x):
        xca = sum([self.ca[i](x) for i in range(3)])
        xpa = [self.pa_modules[i](xca) for i in range(3)]
        res = sum(xpa)
        return res

class Block(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, stride=1):
        super(Block, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
#         self.calayer=CALayer(_in_channels)
#         self.palayer=PALayer(_in_channels)
        self.csa=CSALayer(_in_channels)
        self.conv_1x1=default_conv(_in_channels, in_channels, kernel_size=1)
    def forward(self, x):
        out = self.residual_dense_layers(x)
#         out = self.calayer(out)
#         out = self.palayer(out)
        out = self.csa(out)
        out = self.conv_1x1(out)
        out = out + x
        return out

class Group(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, blocks, kernel_size=3, pool_size=2):
        super(Group, self).__init__()
        self.avg_pool = nn.AvgPool2d(pool_size, stride=pool_size)
        modules = [Block(in_channels, num_dense_layer, growth_rate)]
        for i in range(blocks-1):
            modules.extend([Block(in_channels, num_dense_layer, growth_rate)])
        modules.append(default_conv(in_channels, in_channels, kernel_size))
        modules.append(self.avg_pool)
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += F.interpolate(x, size=[res.size()[-2], res.size()[-1]], mode='bilinear')
        return res
    
class Attention_Pyramid_Feature_Fusion(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(Attention_Pyramid_Feature_Fusion, self).__init__()
        self.gps = gps
        self.in_channels = 16
        self.num_dense_layer = 4
        self.growth_rate = 16
        assert self.gps == 3
        kernel_size = 3
        pool_size = 2
        self.pre_conv = default_conv(3, self.in_channels, kernel_size)
        self.g1= Group(self.in_channels, self.num_dense_layer, self.growth_rate, blocks=blocks, kernel_size=kernel_size, pool_size=pool_size)
        self.g2= Group(self.in_channels, self.num_dense_layer, self.growth_rate, blocks=blocks, kernel_size=kernel_size, pool_size=pool_size)
        self.g3= Group(self.in_channels, self.num_dense_layer, self.growth_rate, blocks=blocks, kernel_size=kernel_size, pool_size=pool_size)
        self.tran_conv = nn.ConvTranspose2d(self.in_channels, self.in_channels, kernel_size=pool_size, stride=pool_size)
        t = []
        for i in range(1, self.gps+1):
            t.append([self.tran_conv for _ in range(i)])
        self.t1, self.t2, self.t3 = nn.Sequential(*t[0]), nn.Sequential(*t[1]), nn.Sequential(*t[2])
#         self.calayer=CALayer(self.in_channels*(self.gps+1))
#         self.palayer=PALayer(self.in_channels*(self.gps+1))
        self.csa=CSALayer(self.in_channels*(self.gps+1))
        self.post_conv = nn.Sequential(default_conv(self.in_channels*(self.gps+1), self.in_channels, kernel_size),
                                       default_conv(self.in_channels, 2, kernel_size))
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x1 = self.pre_conv(x)
        g1=self.g1(x1)
        g2=self.g2(g1)
        g3=self.g3(g2)
#         print(f'g1:{g1.size()}, g2:{g2.size()}, g3:{g3.size()}')
        catg=torch.cat((x1, self.t1(g1), self.t2(g2), self.t3(g3)), 1)
#         ca=self.calayer(catg)
#         pa=self.palayer(ca)
        pa = self.csa(catg)
        res = self.gap(self.post_conv(pa)).squeeze()
        return res


if __name__ == "__main__":
    net=APFF(gps=3,blocks=4)
    print(net)