import torch
from torch import nn
from torch.nn import functional as F
from models.ResNetlw import resnet_instance


class DCNet(nn.Module):
    def __init__(self, n_class, res_depth, pretrained) -> None:
        super().__init__()

        # Build Encoder
        self.encoder = resnet_instance(n_class, depth=res_depth, pretrained=pretrained)
        layer_stride = self.encoder.layer_stride
        layer_stride.reverse()
        # Build Decoder
        input_dims = [2048, 2048, 1024, 512]
        output_dims = [1024, 512, 256, 128]
        self.decoder = []
        for i, (in_channel, out_channel) in enumerate(zip(input_dims, output_dims)):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channel,
                        out_channel,
                        kernel_size=3,
                        stride=layer_stride[i],
                        padding=1,
                        output_padding=layer_stride[i] - 1,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.ModuleList(self.decoder)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                output_dims[-1],
                output_dims[-1],
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.BatchNorm2d(output_dims[-1]),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(output_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        cams, pred, layer_output = self.encoder(x)
        return cams, pred, layer_output

    def decode(self, x, idx):
        return self.decoder[idx](x)

    def forward(self, x):
        cams, pred, layer_output = self.encode(x)
        num_layers = len(layer_output)
        recons = 0
        for i in range(num_layers):
            if i:
                recons = self.decode(
                    torch.cat((layer_output[num_layers - i - 1], recons), dim=1), i
                )
            else:
                recons = self.decode(layer_output[num_layers - i - 1], i)
        recons = self.final_layer(recons)

        return cams, pred, recons
