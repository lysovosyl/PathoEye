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
import numpy as np
from skimage import color
def generate_random_colors(n):
    colors = []
    for _ in range(n):
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
import numpy as np
from PIL import Image
from sklearn.decomposition import NMF

import numpy as np
from skimage import io

def rgb2od(I):
    I = I.astype(np.float64)
    return -np.log((I+1) / 255)
def od2rgb(OD):
    return (255 * np.exp(-OD)).clip(0, 255).astype(np.uint8)
def macenko_normalize(src, tgt, alpha=0.1, beta=0.15):
    # Convert RGB → OD
    OD_src = rgb2od(src)
    OD_tgt = rgb2od(tgt)

    # Filter background pixels
    OD_src = OD_src[~np.any(OD_src < beta, axis=2)]

    # SVD to find stain vectors
    U, s, Vt = np.linalg.svd(OD_src, full_matrices=False)
    stain_vectors = Vt[:2].T

    # Normalize vectors
    stain_vectors /= np.linalg.norm(stain_vectors, axis=0)

    # Compute stain concentrations
    OD_src_flat = rgb2od(src).reshape((-1, 3))
    C = np.dot(OD_src_flat, stain_vectors)

    # Target stain vectors
    OD_tgt = OD_tgt.reshape((-1, 3))
    C_tgt = np.dot(OD_tgt, stain_vectors)

    # Match distributions
    C = (C - C.mean(0)) / C.std(0) * C_tgt.std(0) + C_tgt.mean(0)

    # Reconstruct normalized OD
    OD_norm = np.dot(C, stain_vectors.T).reshape(src.shape)
    return od2rgb(OD_norm)

def reinhard_normalize(src, tgt):
    src_lab = color.rgb2lab(src)
    tgt_lab = color.rgb2lab(tgt)

    for i in range(3):
        src_mean, src_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std()
        tgt_mean, tgt_std = tgt_lab[:,:,i].mean(), tgt_lab[:,:,i].std()
        src_lab[:,:,i] = (src_lab[:,:,i] - src_mean) / src_std * tgt_std + tgt_mean

    return np.clip(color.lab2rgb(src_lab), 0, 1)
class sample_module():
    def __init__(self,seg,size=2048,device='cuda:0',normalize='reinhard',target_img_path=None,target_pixel=[215, 128, 193]):
        self.path = None
        self.device = device
        self.size = size
        self.seg_module = seg
        self.sample_list = []
        self.mask_show_all = None
        self.mask_show_target = None
        self.target_img = None
        if target_img_path != None:
            self.target_img = Image.open(target_img_path)
        self.normalize = normalize
        self.target_pixel = target_pixel

    def sample_object(self,slide):
        self.sample_list = []
        self.tissue_object_list = scan_object(slide)
        for tissue_object in self.tissue_object_list:
            scale_tissue,self.fold = resize_image(tissue_object,2048)
            self.raw_img, mask_target = self.seg_module.seg_image(scale_tissue, device=self.device, minLabels=5,target_pixel=self.target_pixel)
            self.mask_show_target = show_mask(mask_target)
            if self.target_img is None:
                self.target_img = self.raw_img
            mask_target = Image.fromarray(mask_target)
            mask_target = mask_target.resize([tissue_object.shape[1], tissue_object.shape[0]], Image.LANCZOS)
            mask_target = np.array(mask_target)
            contours, hierarchy = cv2.findContours(mask_target, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for pos_list in contours:
                pos_list = pos_list.reshape([-1, 2])
                for index in range(0, len(pos_list), 1024):
                    x, y = pos_list[index]
                    source_img = tissue_object[y - int(self.size / 2):y + int(self.size / 2), x - int(self.size / 2):x + int(self.size / 2)]
                    if self.normalize == 'reinhard':
                        normalized_img = reinhard_normalize(source_img,self.target_img)
                        self.sample_list.append(normalized_img)
                    elif self.normalize == 'macenko':
                        normalized_img = macenko_normalize(source_img, self.target_img)
                        self.sample_list.append(normalized_img)


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

