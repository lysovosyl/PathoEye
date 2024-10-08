#%%
import torch
from PIL import Image
from utils.sample_object import sample_module
from models.patheye import GradCamModel
from utils.segmentation import method_infoseg
import torchvision.transforms as transforms
import openslide
import torch.nn as nn
from skimage.transform import resize
import matplotlib as mpl
import numpy as np
import json
import os
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-input_path',type=str,required=True,help='wsi should be store in this directory')
parse.add_argument('-model_path',type=str,required=True,help='the directory should be generated from train')
parse.add_argument('-save_path',type=str,required=True,help='this directory will save all patch which were sampled from wsi')
parse.add_argument('-device',type=str,required=True)
args = parse.parse_args()

input_path = args.input_path
model_path = args.model_path
save_path = args.save_path
device = args.device


# input_file = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/Zhoujj/20240613类天疱疮/2313359 - 2024-07-25 12.09.41.ndpi'
# model_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/model'
# save_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/one_sample'
# device = 'cuda:1'
#%%
# 读取JSON文件
with open(os.path.join(model_path,'config.json'), 'r', encoding='utf-8') as f:
    model_config = json.load(f)
class_num = model_config['class_num']

#%%
slide = openslide.open_slide(input_path)
model = GradCamModel(class_num).to(device)
model.load_state_dict(torch.load(model_config['model'],weights_only=True))
seg = method_infoseg()
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=mean, std=std)
        ])
#%%
print('sampling')
sampler = sample_module(seg,device=device)
sampler.sample_object(slide)
print('sample complete')
#%%
import os
import matplotlib.pyplot as plt
result = []
for index,img in enumerate(sampler.sample_list):
    img = Image.fromarray(img)
    img = transform(img)
    img = img.to(device).unsqueeze(dim=0)
    outputs,acts = model(img)
    label = torch.zeros_like(outputs)
    label[0, torch.argmax(outputs)] = 1
    loss = nn.CrossEntropyLoss()(outputs, label)
    loss.backward()
    grads = model.get_act_grads()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:, i, :, :] += pooled_grads[i]

    heatmap_j = torch.mean(acts, dim=1).squeeze()
    heatmap_j_max = heatmap_j.max(axis=0)[0]
    heatmap_j /= heatmap_j_max

    heatmap_j = heatmap_j.cpu().detach().numpy()
    heatmap_j = resize(heatmap_j, (512, 512), preserve_range=True)
    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap_j2 = cmap(heatmap_j, alpha=0.2)
    heatmap_j2 = heatmap_j2[:, :, :3]

    inpimg = img
    inpimg = torch.squeeze(inpimg)
    inpimg = torch.permute(inpimg, [1, 2, 0])
    inpimg = inpimg.detach().cpu().numpy()

    heatmap_j2 = heatmap_j2 / np.linalg.norm(heatmap_j2)
    inpimg = inpimg / np.linalg.norm(inpimg)
    superimposed_img = heatmap_j2 * 0.4 + inpimg * 0.6

    superimposed_img = (superimposed_img - np.min(superimposed_img)) * (
            255 / (np.max(superimposed_img) - np.min(superimposed_img)))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    raw_image = (inpimg - np.min(inpimg)) * (255 / (np.max(inpimg) - np.min(inpimg)))
    raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
    plt.imsave(os.path.join(save_path,'{}_raw.png'.format(index)),raw_image)
    plt.imsave(os.path.join(save_path,'{}_heatmap.png'.format(index)),superimposed_img)


    outputs = torch.softmax(outputs, dim=1)
    outputs = outputs.cpu().detach().numpy()
    predict = np.argmax(outputs)
    result.append(predict)
#%%