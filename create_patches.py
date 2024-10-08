#%%
import matplotlib.pyplot as plt
from utils.sample_object import sample_module
from utils.segmentation import method_infoseg
import openslide
import csv
from tqdm import tqdm
import os
import cv2
#%%
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-input_path',type=str,required=True,help='wsi should be store in this directory')
parse.add_argument('-save_path',type=str,required=True,help='this directory will save all patch which were sampled from wsi')
parse.add_argument('-device',type=str,required=True)
args = parse.parse_args()
input_path = args.input_path
save_path = args.save_path
device = args.device

# input_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/patheye/data'
# save_path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/patheye/dataset'
# device = 'cuda:0'
# seg = method_infoseg()
# sampler = sample_module(seg,device=device)


#%%
image_index = {}
for disease in os.listdir(input_path):
    image_index[disease] = {}
    if os.path.exists(os.path.join(save_path,disease)) == False:
        os.makedirs(os.path.join(save_path,disease))
    for disease_index,img in enumerate(tqdm(os.listdir(os.path.join(input_path,disease)))):
        if os.path.exists(os.path.join(save_path, disease,str(disease_index))) == False:
            os.makedirs(os.path.join(save_path, disease,str(disease_index)))
        if os.path.exists(os.path.join(save_path, disease,str(disease_index),'data')) == False:
            os.makedirs(os.path.join(save_path, disease,str(disease_index),'data'))
        image_index[disease][img] = disease_index
        input_file = os.path.join(input_path,disease,img)
        slide = openslide.open_slide(input_file)
        print('sample ing ')
        sampler.sample_object(slide)
        print('sample complete')

        print('imwrite ing',sampler.mask_show_target.shape,sampler.raw_img.shape)
        cv2.imwrite(os.path.join(save_path, disease, str(disease_index), '{}.png'.format('mask_target')),sampler.mask_show_target)
        cv2.imwrite(os.path.join(save_path, disease, str(disease_index), '{}.png'.format('raw_img')),sampler.raw_img)
        print('imwrite ing')
        for img_index,img in enumerate(sampler.sample_list):
            if img_index > 120:
                break
            if img.shape[0] > 1 and img.shape[1] > 1 and img.shape[0] == img.shape[1]:
                cv2.imwrite(os.path.join(save_path,disease,str(disease_index),'data','{}.png'.format(img_index)),img)
#%%

f = open(os.path.join(save_path,'slide_info.csv'),'w')
writer = csv.writer(f)
writer.writerow(['type','slide_name','slide_index'])
for disease in image_index:
    for img in image_index[disease].keys():
        writer.writerow([disease,img,image_index[disease][img]])
f.close()