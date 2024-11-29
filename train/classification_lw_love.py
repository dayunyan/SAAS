import sys, os

sys.path.append("/home/zjj/xjd/SAAS/")
# print(sys.path)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import normalize, resize
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
import torchvision.models as models
import copy
from torcheval.metrics import MultilabelAccuracy

from models.ResNetlw import resnet_instance
from models.loser import CAMRefineLoss
from PIL import Image
from datasets.loveda import LoveDAClassification
from utils import args, config

MODE = args.mode
device = torch.device(args.device)


def train():
    train_dataset = LoveDAClassification(
        image_dir=MODE.train_dir, mask_dir=config.train_mask_dir, transform=args.T
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize1,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataset = LoveDAClassification(
        image_dir=MODE.val_dir, mask_dir=config.val_mask_dir, transform=args.T
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize1,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # test_dataset = datasets.ImageFolder(root=config.test_dir,transform=data_transform)
    # test_dataloader = DataLoader(dataset=test_dataset,
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               num_workers=4)

    print("train_dataset: {}".format(len(train_dataset)))
    print("val_dataset: {}".format(len(val_dataset)))
    # print('test_dataset: {}'.format(len(test_dataset)))

    # 载入训练模型
    model_name = f"{args.cls_model}_{MODE.name}"
    if args.cls_model == "resnet":
        model = resnet_instance(n_class=args.num_classes, pretrained=True)
    # print(tmodel)
    model.to(device)

    # 优化方法、损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = [nn.MultiLabelSoftMarginLoss()]
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    ## 训练
    num_epoch = config.num_epoch
    # 训练日志保存
    logfile_dir = config.logfile_dir

    acc_best_wts = model.state_dict()
    best_train_acc = 0
    best_acc = 0
    iter_count = 0
    a = 0
    for epoch in range(num_epoch):
        train_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total = 0

        val_loss = 0.0
        val_acc = 0.0
        val_correct = 0
        val_total = 0
        for i, sample_batch in enumerate(train_dataloader):
            # print(i)
            inputs = sample_batch["image"].to(device)
            labels = sample_batch["label"].to(device)
            # print(labels)

            # 模型设置为train
            model.train()

            # forward
            cams, outputs = model(inputs)

            # print(outputs.size())
            # loss
            # print(f'cross_entropy:{criterion[0](outputs, labels)}, cam:{criterion[1](cams, inputs)}')
            loss = criterion[0](outputs, labels)  # + a*criterion[1](cams, inputs)

            # forward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            outputs = F.sigmoid(outputs)
            train_correct += (
                ((outputs > 0.5).float() * labels).sum().item()
            )  #    (torch.max(outputs, 1)[1] == labels).sum().item()
            train_total += labels.sum().item()  # labels.size(0)

            # print('iter:{}'.format(i))

            if i % 10 == 9:
                # print(i)
                for sample_batch in val_dataloader:
                    # inputs = sample_batch[0].to(device)
                    # labels = sample_batch[1].to(device)
                    inputs = sample_batch["image"].to(device)
                    labels = sample_batch["label"].to(device)

                    model.eval()
                    with torch.no_grad():
                        cams, outputs = model(inputs)
                        loss = criterion[0](
                            outputs, labels
                        )  # + a*criterion[1](cams, inputs)
                        _, prediction = torch.max(outputs, 1)
                        outputs = F.sigmoid(outputs)
                        val_correct += (
                            ((outputs > 0.5).float() * labels).sum().item()
                        )  # ((labels == prediction).sum()).item()
                        val_total += labels.sum().item()  # inputs.size(0)
                        val_loss += loss.item()

                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                print(
                    "[{},{}] train_loss = {:.5f} train_acc = {:.5f} val_loss = {:.5f} val_acc = {:.5f}".format(
                        epoch + 1,
                        i + 1,
                        train_loss / 100,
                        train_correct / train_total,
                        val_loss / len(val_dataloader),
                        val_correct / val_total,
                    )
                )
                if val_acc >= best_acc:
                    if val_acc > best_acc or (
                        val_acc == best_acc and train_acc > best_train_acc
                    ):
                        best_train_acc = train_acc
                        best_acc = val_acc
                        acc_best_wts = copy.deepcopy(model.state_dict())

                with open(logfile_dir + "train_loss.txt", "a") as f:
                    f.write(str(train_loss / 100) + "\n")
                with open(logfile_dir + "train_acc.txt", "a") as f:
                    f.write(str(train_correct / train_total) + "\n")
                with open(logfile_dir + "val_loss.txt", "a") as f:
                    f.write(str(val_loss / len(val_dataloader)) + "\n")
                with open(logfile_dir + "val_acc.txt", "a") as f:
                    f.write(str(val_correct / val_total) + "\n")

                iter_count += 200

                train_loss = 0.0
                train_total = 0
                train_correct = 0
                val_correct = 0
                val_total = 0
                val_loss = 0
        scheduler.step()
        print(optimizer.state_dict()["param_groups"][0]["lr"])

    print("Train finish!")
    # 保存模型
    model_path = MODE.model_path
    # with open(model_file+'/model_squeezenet_teeth_1.pth','a') as f:
    #     torch.save(acc_best_wts,f)
    torch.save(acc_best_wts, model_path)
    print("Model save ok!")


def test():
    val_dataset = LoveDAClassification(
        image_dir=MODE.val_dir, mask_dir=config.val_mask_dir, transform=args.T
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # 测试

    model_name = f"{args.cls_model}_{MODE.name}"
    model = resnet_instance(n_class=args.num_classes, pretrained=True)
    #     print(model)
    # model.to(device)
    model.load_state_dict(torch.load(MODE.model_path, map_location=device))
    # print(model)
    model.to(device)
    model.eval()

    val_correct = 0
    val_total = 0
    acc = 0.0
    tp = fp = fn = tn = 0

    for i, sample_batch in enumerate(val_dataloader):
        # inputs = sample_batch[0].to(device)
        # labels = sample_batch[1].to(device)
        inputs = sample_batch["image"].to(device)
        labels = sample_batch["label"].to(device)

        with torch.no_grad():
            _, outputs = model(inputs)

            outputs = F.sigmoid(outputs)

            tp += torch.sum(outputs * labels, dim=0)
            fp += torch.sum(outputs * (1 - labels), dim=0)
            fn += torch.sum((1 - outputs) * labels, dim=0)
            tn += torch.sum((1 - outputs) * (1 - labels), dim=0)
            # val_correct += (
            #     ((outputs > 0.5).float() * labels).sum().item()
            # )  # ((labels == prediction).sum()).item()
            # val_total += labels.sum().item()  # inputs.size(0)
            if i % 500 == 0:
                for b in range(inputs.size(0)):
                    print(
                        f"Batch no.{i} --- predictions:{(outputs[b] > 0.5).float()} \n labels:{labels[b]} \n"
                    )
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    spec = tn / (tn + fp)
    fscore = (2 * pre * recall) / (pre + recall)
    print(
        "\n[TEST]\n".join(
            "acc = {:.5f}, precision = {:.5f}, Recall = {:.5f}, Specificity = {:.5f}, F1-score = {:.5f}".format(
                *vals
            )
            for vals in zip(acc, pre, recall, spec, fscore)
        )
    )


if __name__ == "__main__":
    # train()
    test()
