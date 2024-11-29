import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineLoss(torch.nn.Module):
    def __init__(self, th=0.5, alpha=0.9, beta=0.1, r=8):
        super(RefineLoss, self).__init__()

        self.th = th
        self.a = alpha
        self.b = beta
        self.r = r  # 半径

    def forward(self, cam, img):
        assert cam.dim() == 4
        assert img.dim() == 4
        b, c, h, w = img.size()
        b1, c1, h1, w1 = cam.size()
        if not c1 == 1:
            cam = cam[:, 1:, :, :]
            c1 = 1
        if not (h == h1 and w == w1):
            cam = nn.Upsample(size=(h, w), mode="bilinear")(cam)
        # cam = (cam-torch.min(cam, dim=0)[0])/(torch.max(cam, dim=0)[0]-torch.min(cam, dim=0)[0])
        cam = torch.where(cam > self.th, 1, 0).view(b1, c1 * h * w)  # [b, 1*256*256]
        img = img.permute(0, 2, 3, 1).view(b, h * w, c)
        # print(f'{torch.min(img)}, {torch.max(img)}')

        loss_inter = []
        loss_cross = []
        for i in range(b):
            cls_1 = img[i].index_select(0, cam[i])
            cls_0 = img[i].index_select(0, 1 - cam[i])
            num_1 = cls_1.size(0)
            num_0 = cls_0.size(0)
            index_1 = torch.LongTensor(random.sample(range(num_1), num_1 // 8))
            index_0 = torch.LongTensor(random.sample(range(num_0), num_0 // 8))
            pro_1 = torch.mean(cls_1[index_1, :], dim=0)
            pro_0 = torch.mean(cls_0[index_0, :], dim=0)
            # print(f'pro_1:{pro_1}, pro_0:{pro_0}')
            loss_inter.append(
                F.mse_loss(cls_1, pro_1.unsqueeze(0).repeat(num_1, 1))
                + F.mse_loss(cls_0, pro_0.unsqueeze(0).repeat(num_0, 1))
            )
            loss_cross.append(
                F.mse_loss(pro_0, pro_1)
                / (
                    F.mse_loss(cls_1, pro_0.unsqueeze(0).repeat(num_1, 1))
                    + F.mse_loss(cls_0, pro_1.unsqueeze(0).repeat(num_0, 1))
                    + 1e-8
                )
            )
            # print(f'{loss_cross[-1]}')
        # print(f'loss_inter:{sum(loss_inter)}, loss_cross:{sum(loss_cross)}')
        return (self.a * sum(loss_inter) + self.b * sum(loss_cross)) / b


class CAMRefineLoss(torch.nn.Module):
    def __init__(self, th=0.5, alpha=1.0, beta=0.5, r=4):
        super(CAMRefineLoss, self).__init__()

        self.th = th
        self.a = alpha
        self.b = beta
        self.r = r  # 采样比例

    def forward(self, cam_cln, cam_haz, img_haz):
        assert cam_cln.dim() == 4
        assert cam_haz.dim() == 4
        assert img_haz.dim() == 4
        b, c, h, w = img_haz.size()
        b1, c1, h1, w1 = cam_cln.size()
        if not c1 == 1:
            cam_cln = cam_cln[:, 1:, :, :]
            cam_haz = cam_haz[:, 1:, :, :]
            c1 = 1
        if not (h == h1 and w == w1):
            cam_cln = nn.Upsample(size=(h, w), mode="bilinear")(cam_cln)
            cam_haz = nn.Upsample(size=(h, w), mode="bilinear")(cam_haz)
        cam_cln = self.otsu_thresholding(cam_cln * 255).repeat(1, 3, 1, 1)
        cam_haz = self.otsu_thresholding(cam_haz * 255).repeat(1, 3, 1, 1)
        # cam_cln = (cam_cln > self.th).repeat(1, 3, 1, 1)
        # cam_haz = (cam_haz > self.th).repeat(1, 3, 1, 1)
        # img_fore_cln = img_haz[~cam_cln]
        # img_back_cln = img_haz[~cam_cln].view(b, c, -1)
        # img_fore_haz = img_haz[cam_haz].view(b, c, -1)
        # img_back_haz = img_haz[~cam_haz].view(b, c, -1)
        img_fore_cln, img_back_cln, img_fore_haz, img_back_haz = [
            img_haz.clone() for _ in range(4)
        ]
        img_fore_cln[~cam_cln] = 0
        img_back_cln[cam_cln] = 0
        img_fore_haz[~cam_haz] = 0
        img_back_haz[cam_haz] = 0

        range_val = (
            (
                torch.min(img_haz) + 0.001
                if torch.min(img_haz) == 0
                else torch.min(img_haz)
            ),
            torch.max(img_haz),
        )
        # print(range_val)
        ce_positive = self.cross_entropy_loss(
            img_fore_cln, img_fore_haz, range_val
        ) + self.cross_entropy_loss(img_back_cln, img_back_haz, range_val)
        ce_negative = -(
            self.cross_entropy_loss(img_fore_cln, img_back_haz, range_val)
            + self.cross_entropy_loss(img_back_cln, img_fore_haz, range_val)
        )
        # print(f"ce_positive:{ce_positive}, ce_negative:{ce_negative}")

        return self.a * ce_positive + self.b * ce_negative
        # # cam = (cam-torch.min(cam, dim=0)[0])/(torch.max(cam, dim=0)[0]-torch.min(cam, dim=0)[0])
        # cam_cln = torch.where(cam_cln > self.th, 1, 0).view(
        #     b1, c1 * h * w
        # )  # [b, 1*256*256]
        # cam_haz = torch.where(cam_haz > self.th, 1, 0).view(b1, c1 * h * w)
        # img_haz = img_haz.permute(0, 2, 3, 1).view(b, h * w, c)
        # # print(f'{torch.min(img)}, {torch.max(img)}')

        # loss_inter = []
        # loss_cross = []
        # for i in range(b):
        #     # protype
        #     cls_1 = img_haz[i].index_select(0, cam_cln[i])
        #     cls_0 = img_haz[i].index_select(0, 1 - cam_cln[i])
        #     num_1 = cls_1.size(0)
        #     num_0 = cls_0.size(0)
        #     index_1 = torch.LongTensor(random.sample(range(num_1), num_1 // self.r))
        #     index_0 = torch.LongTensor(random.sample(range(num_0), num_0 // self.r))
        #     pro_1 = torch.mean(cls_1[index_1, :], dim=0)
        #     pro_0 = torch.mean(cls_0[index_0, :], dim=0)

        #     # predict
        #     cls_1 = img_haz[i].index_select(0, cam_haz[i])
        #     cls_0 = img_haz[i].index_select(0, 1 - cam_haz[i])
        #     num_1 = cls_1.size(0)
        #     num_0 = cls_0.size(0)
        #     index_1 = torch.LongTensor(random.sample(range(num_1), num_1 // self.r))
        #     index_0 = torch.LongTensor(random.sample(range(num_0), num_0 // self.r))
        #     pro_1_hat = torch.mean(cls_1[index_1, :], dim=0)
        #     pro_0_hat = torch.mean(cls_0[index_0, :], dim=0)

        #     # print(f'pro_1:{pro_1}, pro_0:{pro_0}')
        #     loss_inter.append(
        #         F.mse_loss(pro_1, pro_1_hat) + F.mse_loss(pro_0, pro_0_hat)
        #     )
        #     loss_cross.append(
        #         F.mse_loss(pro_0, pro_1)
        #         / (F.mse_loss(pro_1, pro_0_hat) + F.mse_loss(pro_0, pro_1_hat) + 1e-8)
        #     )
        #     # print(f'{loss_cross[-1]}')
        # # print(f'loss_inter:{sum(loss_inter)}, loss_cross:{sum(loss_cross)}')
        # return (self.a * sum(loss_inter) + self.b * sum(loss_cross)) / b

    def calculate_histogram(self, image, range_val: tuple):
        # 将RGB图像转换为概率分布
        # 计算每个通道的直方图
        bins = 256
        min_val = range_val[0]
        max_val = range_val[1]
        hist_red = torch.histc(
            image[:, 0, :, :].contiguous().view(-1), bins=bins, min=min_val, max=max_val
        )
        hist_green = torch.histc(
            image[:, 1, :, :].contiguous().view(-1), bins=bins, min=min_val, max=max_val
        )
        hist_blue = torch.histc(
            image[:, 2, :, :].contiguous().view(-1), bins=bins, min=min_val, max=max_val
        )

        # 组合三个通道的直方图
        histogram = torch.stack((hist_red, hist_green, hist_blue), dim=0)

        # 转换直方图为概率分布
        probability_distribution = histogram / torch.sum(histogram)
        # probability_distribution = probability_distribution[
        #     probability_distribution >= 0
        # ]

        # 计算信息熵
        # entropy = -torch.sum(
        #     probability_distribution * torch.log2(probability_distribution)
        # )
        return probability_distribution

    def cross_entropy_loss(self, image1, image2, range_val: tuple):
        # 计算两张图像的信息熵
        prob_dist1 = self.calculate_histogram(image1, range_val)
        prob_dist2 = self.calculate_histogram(image2, range_val)

        # 为了避免计算log(0)，给概率分布加上一个小的epsilon
        epsilon = 1e-10
        prob_dist1 = prob_dist1.clamp(min=epsilon)
        prob_dist2 = prob_dist2.clamp(min=epsilon)

        # 计算交叉熵
        cross_entropy = -torch.sum(prob_dist1 * torch.log(prob_dist2))
        return cross_entropy

    def otsu_thresholding(self, tensor):
        # Assuming the input tensor shape is [B, C, H, W]
        # Flatten the tensor to shape [B, C, H*W]
        flat_tensor = tensor.view(tensor.size(0), tensor.size(1), -1)

        # Calculate the histogram for each channel in each batch
        hist = torch.stack(
            [
                torch.stack([torch.histc(c, bins=256, min=0, max=255) for c in b])
                for b in flat_tensor
            ]
        )

        # Calculate the probability of each intensity
        prob = hist / hist.sum(dim=2, keepdim=True)

        # Calculate the cumulative sum of probabilities and the cumulative mean
        cum_prob = torch.cumsum(prob, dim=2)
        cum_mean = torch.cumsum(
            prob
            * torch.arange(256, device=flat_tensor.device).repeat(
                tensor.size(0), tensor.size(1), 1
            ),
            dim=2,
        )

        # Calculate the global mean
        global_mean = cum_mean[:, :, -1].unsqueeze(2).repeat(1, 1, 256)

        # Calculate the between-class variance
        numerator = (global_mean * cum_prob - cum_mean) ** 2
        denominator = cum_prob * (1.0 - cum_prob)
        between_class_variance = numerator / denominator

        # Find the threshold that maximizes the between-class variance
        _, threshold = torch.max(between_class_variance, dim=2)

        # Apply the threshold to binarize the image
        binary_tensor = tensor > threshold.view(
            threshold.size(0), threshold.size(1), 1, 1
        )

        return binary_tensor


class BackwardLoss(nn.Module):
    def __init__(
        self, model: nn.Module, threshold=0.7, device=torch.device("cpu")
    ) -> None:
        super(BackwardLoss, self).__init__()

        self.model = copy.deepcopy(model)
        self.device = device
        self.model.to(device)
        # self.th = threshold
        self.th = [0.5, 0.7]

    def forward(self, cam, img):
        assert cam.dim() == 4
        assert img.dim() == 4
        b, c, h, w = img.size()
        b1, c1, h1, w1 = cam.size()
        if not c1 == 1:
            cam = cam[:, 1:, :, :]
            c1 = 1
        if not (h == h1 and w == w1):
            cam = nn.Upsample(size=(h, w), mode="bilinear")(cam)
        loss = []
        for thre in self.th:
            sa = (cam > thre).repeat(1, 3, 1, 1)

            img_fore, img_back = [img.clone() for _ in range(2)]
            img_fore[~sa] = 0
            img_back[sa] = 0
            _, output_fore = self.model(img_fore)
            _, output_back = self.model(img_back)
            label_fore = torch.eye(2)[torch.ones((b,)).long(), :].to(self.device)
            label_back = torch.eye(2)[torch.zeros((b,)).long(), :].to(self.device)
            loss_fore = nn.BCELoss()(torch.sigmoid(output_fore), label_fore)
            loss_back = nn.BCELoss()(torch.sigmoid(output_back), label_back)
            loss.append(loss_fore + 0.1 * loss_back)
            print(f"loss_fore: {loss_fore}, loss_back: {loss_back}")

        return sum(loss) / len(self.th)
