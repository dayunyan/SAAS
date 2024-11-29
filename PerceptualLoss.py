import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(torch.sqrt(F.mse_loss(dehaze_feature, gt_feature)))
            # if torch.isnan(loss[-1]):
            #     print(f'{torch.any(torch.isnan(dehaze_feature))}, {torch.any(torch.isnan(gt_feature))}')
            #     if torch.any(torch.isnan(dehaze_feature)):
            #         print(torch.any(torch.isnan(dehaze)))
            #     if torch.any(torch.isnan(gt_feature)):
            #         print(torch.any(torch.isnan(gt)))
        # print(loss)
        return sum(loss)/len(loss)