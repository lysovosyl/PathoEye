#%%
import os
import torch
import torch.nn as nn
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset
from torch.utils.data import dataloader
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
from torchvision import models
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
import argparse

# model_path = '/mnt/dfc_data2/project/linyusen/project/19_skin_feature/deepmodel/reenet50_60_20_不重合区域模型/model.pth'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/cut_image512_deeplearning/cam_result'
# data1_apth = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/deeplearning_dataset/test/20-29'
# data2_apth = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/deeplearning_dataset/test/60-69'

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('-model_path', help='model file save path result from 05_train_DCNN.py',required=True)
parser.add_argument('-save_path' , help='The path where the result willed be saved',required=True)
parser.add_argument('-data1_path', help='Data files path',required=True)
parser.add_argument('-data2_path', help='The path where the result willed be saved',required=True)
args = parser.parse_args()

model_path = args.model_path
save_path  = args.save_path
data1_path = args.data1_path
data2_path = args.data2_path




test_dataset = LoadData(input_path1 = data1_path,input_path2 = data2_path)
test_dataloader = dataloader.DataLoader (dataset=test_dataset,batch_size=1,shuffle=True)

gcmodel = GradCamModel().to('cuda:0')
gcmodel.load_state_dict(torch.load(model_path))
gcmodel.eval()
#%%
for index,data in enumerate(test_dataloader):
    print(index)
    inpimg = data[0][0:1]
    label  = data[1][0:1]
    inpimg = inpimg.float()
    inpimg = inpimg.to('cuda:0')
    out, acts = gcmodel(inpimg)
    acts = acts.detach().cpu()

    loss = nn.CrossEntropyLoss()(out, label.to('cuda:0'))
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:, i, :, :] += pooled_grads[i]

    heatmap_j = torch.mean(acts, dim=1).squeeze()
    heatmap_j_max = heatmap_j.max(axis=0)[0]
    heatmap_j /= heatmap_j_max

    heatmap_j = heatmap_j.cpu().numpy()
    heatmap_j = resize(heatmap_j,(512,512),preserve_range=True)
    cmap = mpl.cm.get_cmap('jet',256)
    heatmap_j2 = cmap(heatmap_j,alpha = 0.2)
    heatmap_j2 = heatmap_j2[:,:,:3]
    inpimg = torch.squeeze(inpimg)
    inpimg = torch.permute(inpimg,[1,2,0])
    img    = inpimg.cpu().numpy()
    img = img/255
    heatmap = 0.6*img + 0.4*heatmap_j2
    plt.figure()
    plt.imshow(heatmap)
    plt.savefig(os.path.join(save_path,str(index)+'.png'))
    plt.close()

