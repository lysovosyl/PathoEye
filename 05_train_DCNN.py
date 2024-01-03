import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset
from torch.utils.data import dataloader
from torchvision import models

class LoadData(dataset.Dataset):
    def __init__(self,input_path1,input_path2):
        super(LoadData, self).__init__()

        num = 500
        self.image = []
        self.label = []
        for index, file in enumerate(os.listdir(input_path1)):
            if index == num:
                break
            im = plt.imread(os.path.join(input_path1, file))
            im = np.array(im)
            im = im.transpose((2,0,1))
            self.image.append(im)
            self.label.append(np.array([0.0,1.0]))

        for index, file in enumerate(os.listdir(input_path2)):
            if index == num:
                break
            im = plt.imread(os.path.join(input_path2, file))
            im = np.array(im)
            im = im.transpose((2, 0, 1))
            self.image.append(im)
            self.label.append(np.array([1.0,0.0]))


    def __getitem__(self,index):
        img = self.image[index]
        label = self.label[index]


        return img, label


    def __len__(self):
        return len(self.image)

class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        self.pretrained = models.resnet50(pretrained=True)

        resnet50 = models.resnet50(pretrained=True)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 2),nn.LogSoftmax(dim=1))
        self.pretrained =resnet50
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

#%%

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('-train_path1', help='The path, which were generate by 04_make_deeplearning_dataset.py',required=True)
parser.add_argument('-train_path2' , help='The path, which were generate by 04_make_deeplearning_dataset.py',required=True)
parser.add_argument('-test_path1', help='The path, which were generate by 04_make_deeplearning_dataset.py',required=True)
parser.add_argument('-test_path2', help='The path, which were generate by 04_make_deeplearning_dataset.py',required=True)
parser.add_argument('-save_path', help='The path to save model',required=True)
parser.add_argument('-save_file_name', help='The path where the result willed be saved',default='model.pth')
args = parser.parse_args()


train_path1 = args.train_path1
train_path2 = args.train_path2

test_path1  = args.test_path1
test_path2  = args.test_path2

save_path   = args.save_path
save_file_name = args.save_file_name



if os.path.exists(save_path) == False:
    os.makedirs(save_path)
save_path = os.path.join(save_path,save_file_name)
#%%
model = GradCamModel()
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, )
model = model.to('cuda:0')
# 设置网络训练的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 20
train_dataset = LoadData(
    input_path1 = train_path1,
    input_path2 = train_path2)
train_dataloader = dataloader.DataLoader (
    dataset=train_dataset,
    batch_size=20,
    shuffle=True
    )
test_dataset = LoadData(
    input_path1 = test_path1,
    input_path2 = test_path2)
test_dataloader = dataloader.DataLoader (
    dataset=test_dataset,
    batch_size=10,
    shuffle=True
    )
#%%
import torchmetrics
metric = torchmetrics.Accuracy(task='binary').to('cuda:0')
validation_loss_history = []
train_loss_history = []


validation_acc_history = []
train_acc_history = []
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))
    model.train()
    loss_history = []
    acc_history = []
    metric.reset()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.float()
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs,select_img = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.float()
        targets = targets.int()
        metric(outputs, targets)
        total_train_step = total_train_step + 1
        loss_history.append(loss.item())
        acc_history.append(metric.compute().cpu().numpy())
    print("Loss: {},train acc:{}".format(np.average(loss_history), str(np.around(metric.compute().cpu().numpy(), 4))))
    train_loss_history.append(np.average(loss_history))
    train_acc_history.append(np.average(acc_history))

    model.eval()
    loss_history = []
    acc_history = []
    metric.reset()
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.float()
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs, select_img = model(imgs)
        loss = loss_fn(outputs, targets)

        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.float()
        targets = targets.int()
        metric(outputs, targets)
        loss_history.append(loss.item())
        acc_history.append(metric.compute().cpu().numpy())
    print("Loss: {},test acc:{}".format(np.average(loss_history), str(np.around(metric.compute().cpu().numpy(), 4))))
    validation_loss_history.append(np.average(loss_history))
    validation_acc_history.append(np.average(acc_history))
#%%

plt.figure(figsize=(8, 6))
plt.plot(train_loss_history, marker='o', linestyle='-')
plt.plot(validation_loss_history, marker='o', linestyle='-')
plt.legend(['train','test'])
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(save_path,'Loss.png'))

plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, marker='o', linestyle='-')
plt.plot(validation_acc_history, marker='o', linestyle='-')
plt.legend(['train','test'])
plt.title('Acc Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.grid(True)
plt.savefig(os.path.join(save_path,'Acc.png'))

torch.save(model.state_dict(), save_path)
