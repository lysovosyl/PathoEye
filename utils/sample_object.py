#%%
import sys

import matplotlib.pyplot as plt

sys.path.append('../')
import openslide
from PIL import Image
import cv2
import numpy as np
import random
from utils.segmentation import method_infoseg
from utils.search_object import scan_object,resize_image
def generate_random_colors(n):
    colors = []
    for _ in range(n):
        # 随机生成 RGB 值，每个值在 0-255 之间
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors
def show_mask(mask):
    color_num = len(np.unique(mask))
    color_list = generate_random_colors(color_num)
    mask_color = np.zeros([mask.shape[0],mask.shape[1],3])
    for index,label in enumerate(np.unique(mask)):
        mask_color[mask==label] = color_list[index]
    mask_color = mask_color.astype(int)
    return mask_color.astype(np.uint8)


def resize_image(img:np.array,size=2048):
    img = Image.fromarray(img)
    width,hight = img.size[:2]
    fold = max(img.size[:2])/size
    new_size = (int(width / fold),int(hight / fold))
    resized_image = img.resize(new_size)
    img = np.array(resized_image)
    return img,fold
#%%
class sample_module():
    def __init__(self,seg,size=2048,device='cuda:0'):
        self.path = None
        self.device = device
        self.size = size
        self.seg_module = seg
        self.sample_list = []
        self.mask_show_all = None
        self.mask_show_target = None
    def sample_object(self,slide):
        self.sample_list = []
        self.tissue_object_list = scan_object(slide)
        for tissue_object in self.tissue_object_list:
            scale_tissue,self.fold = resize_image(tissue_object,2048)
            self.raw_img, mask_target = self.seg_module.seg_image(scale_tissue, device=self.device, minLabels=5)
            self.mask_show_target = show_mask(mask_target)
            mask_target = Image.fromarray(mask_target)
            mask_target = mask_target.resize([tissue_object.shape[1], tissue_object.shape[0]], Image.LANCZOS)
            mask_target = np.array(mask_target)
            contours, hierarchy = cv2.findContours(mask_target, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for pos_list in contours:
                pos_list = pos_list.reshape([-1, 2])
                for index in range(0, len(pos_list), 1024):
                    x, y = pos_list[index]
                    img = tissue_object[y - int(self.size / 2):y + int(self.size / 2), x - int(self.size / 2):x + int(self.size / 2)]
                    self.sample_list.append(img)

if __name__ == '__main__':
    path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/Zhoujj/healthy2/GTEX-XQ8I-0426.svs'
    source = openslide.OpenSlide(path)
    seg_method = method_infoseg()
    sampler = sample_module(seg_method)
    sampler.sample_object(source)
    for index,img in enumerate(sampler.sample_list):
        if img.shape[0] > 1 and img.shape[1] > 1:
            img,fold = resize_image(img, 512)
            plt.imsave('/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/test_sample/{}.png'.format(index),img)

