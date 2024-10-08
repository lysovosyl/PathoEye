#%%
from torch.utils.data import dataloader
import os
import torch
import torch.nn as nn
import numpy as np
from models.patheye import GradCamModel
from datasets.dataloader import LoadData,class_index
import torchmetrics
import json
import csv
from tqdm import tqdm
#%%
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-train_path',type=str,required=True,help='this directory should be generator from create_patches.py')
parse.add_argument('-val_path',type=str,required=True,help='this directory should be generator from create_patches.py')
parse.add_argument('-save_path',type=str,required=True,help='this directory will save the config of the model and its weight')
parse.add_argument('-device',type=str,default='cuda:0')
parse.add_argument('-epochs',type=int,default=20)
parse.add_argument('-learning_rate',type=int,default=0.01)
parse.add_argument('-batch_size',type=int,default=9)
parse.add_argument('-num_workers',type=int,default=8,help='the number of thread that are used to load data')
args = parse.parse_args()

train_path = args.train_path
val_path = args.val_path
save_path = args.save_path
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
device = args.device
num_workers = args.num_workers


# train_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/train3'
# test_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/test3'
# save_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/model'
# epochs = 1
# learning_rate = 0.01
# batch_size = 9
# device = 'cuda:1'
label_index = class_index(train_path)
class_num = len(label_index)
model = GradCamModel(class_num).to(device)
train_data = LoadData(train_path, class_num)
val_data = LoadData(val_path, class_num)
train_dataloader = dataloader.DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=num_workers)
val_dataloader = dataloader.DataLoader(val_data, batch_size=batch_size,shuffle=True,num_workers=num_workers)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
metric = torchmetrics.Accuracy().to(device)
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

validation_loss_history = []
train_loss_history = []
validation_acc_history = []
train_acc_history = []

for i in range(epochs):
    model.train()
    loss_history = []
    acc_history = []
    metric.reset()
    progress_bar = tqdm(train_dataloader,desc=f"Training Epoch:{i+1}/{epochs} ")
    for data in progress_bar:
        imgs, targets = data
        imgs = torch.cat(imgs, dim=0)
        targets = torch.cat(targets, dim=0)

        imgs = imgs.float()

        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs, select_img = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.float()
        targets = targets.int()
        metric(outputs, targets)
        loss_history.append(loss.item())
        acc_history.append(metric.compute().cpu().numpy())
        loss_value = np.average(loss_history)
        accuracy = metric.compute().cpu().numpy()
        progress_bar.set_postfix({"Loss": f"{loss_value:.4f}", "Acc": f"{accuracy:.4f}"})

    train_loss_history.append(np.average(loss_history))
    train_acc_history.append(np.average(acc_history))

    model.eval()
    loss_history = []
    acc_history = []
    metric.reset()
    progress_bar = tqdm(val_dataloader, desc=f"Val Epoch:{i+1}/{epochs} ")
    for data in progress_bar:
        imgs, targets = data
        imgs = torch.cat(imgs, dim=0)
        targets = torch.cat(targets, dim=0)
        imgs = imgs.float()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs, select_img = model(imgs)

        loss = loss_fn(outputs, targets)

        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.float()
        targets = targets.int()
        metric(outputs, targets)
        loss_history.append(loss.item())
        acc_history.append(metric.compute().cpu().numpy())
        loss_value = np.average(loss_history)
        accuracy = metric.compute().cpu().numpy()
        loss_value = np.average(loss_history)
        accuracy = metric.compute().cpu().numpy()
        progress_bar.set_postfix({"Loss": f"{loss_value:.4f}", "Acc": f"{accuracy:.4f}"})
    validation_loss_history.append(np.average(loss_history))
    validation_acc_history.append(np.average(acc_history))


if os.path.exists(save_path) == False:
    os.makedirs(save_path)
torch.save(model.state_dict(), os.path.join(save_path,'model.pth'))
config = {
    'model':os.path.join(save_path,'model.pth'),
    'label':label_index,
    'class_num':class_num
}

with open(os.path.join(save_path,'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# with open(os.path.join(save_path,'train_loss.csv'), 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(train_loss_history)
# with open(os.path.join(save_path,'train_acc.csv'), 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(train_acc_history)
#
# with open(os.path.join(save_path,'val_loss.csv'), 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(validation_loss_history)
# with open(os.path.join(save_path,'val_acc.csv'), 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(validation_acc_history)