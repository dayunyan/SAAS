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

MODE = args.mode

train_dataset = datasets.ImageFolder(root=MODE.train_dir, transform=args.T)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batchsize1,
    shuffle=True,
    num_workers=args.num_workers,
)

val_dataset = datasets.ImageFolder(root=MODE.val_dir, transform=args.T)
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


device = torch.device(args.device)

# 载入训练模型
model_name = f"{args.cls_model}_{MODE.name}"
if args.cls_model == "resnet":
    model = resnet_instance(n_class=args.num_classes, pretrained=True)
# print(tmodel)
model.to(device)

# 优化方法、损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = [nn.CrossEntropyLoss()]
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
        inputs = sample_batch[0].to(device)
        labels = sample_batch[1].to(device)
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
        train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        train_total += labels.size(0)

        # print('iter:{}'.format(i))

        if i % 10 == 9:
            # print(i)
            for sample_batch in val_dataloader:
                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                model.eval()
                with torch.no_grad():
                    cams, outputs = model(inputs)
                    loss = criterion[0](
                        outputs, labels
                    )  # + a*criterion[1](cams, inputs)
                    _, prediction = torch.max(outputs, 1)
                    val_correct += ((labels == prediction).sum()).item()
                    val_total += inputs.size(0)
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
model_path = MODE.model_nonconsist_path
# with open(model_file+'/model_squeezenet_teeth_1.pth','a') as f:
#     torch.save(acc_best_wts,f)
torch.save(acc_best_wts, model_path)
print("Model save ok!")


# 测试

model_name = f"{args.cls_model}_{MODE.name}"
model = resnet_instance(n_class=args.num_classes, pretrained=True)
#     print(model)
# model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
# print(model)
model.eval()

correct = 0
total = 0
acc = 0.0
for file in os.listdir(MODE.test_dir):
    if "checkpoints" in file:
        continue
    img = Image.open(f"{MODE.test_dir}{file}")
    inputs = normalize(
        torch.tensor(np.array(resize(img, (256, 256))).transpose(2, 0, 1) / 255.0),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    _, outputs = model(inputs.unsqueeze(0).to(torch.float32))
    print(outputs)
    _, prediction = torch.max(outputs, 1)
    print(prediction)
#     correct += (labels == prediction).sum().item()
#     total += labels.size(0)

# acc = correct / total
# print('test finish, total:{}, correct:{}, acc:{:.3f}'.format(total, correct, acc))