import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


target_pixel = np.array([[134.71875,52.84375,99.375]])
# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/cut_image/512'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_512'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True,help='the dir of segmentation image(512*512) result from 01_cut_image.py')
parser.add_argument('-save_path',type=str,required=True,help='the dir of save filter result(dir path)')
args = parser.parse_args()

input_path = args.input_path
save_path = args.save_path
#%%

distance_threshold = 20

for index,dir in enumerate(os.listdir(input_path)):
    if os.path.exists(os.path.join(save_path,dir)) == False:
        os.makedirs(os.path.join(save_path,dir))

    if os.path.exists(os.path.join(save_path,dir,'jpg')) == False:
        os.makedirs(os.path.join(save_path,dir,'jpg'))
    if os.path.exists(os.path.join(save_path,dir,'image')) == False:
        os.makedirs(os.path.join(save_path,dir,'image'))
    if os.path.exists(os.path.join(save_path,dir,'mask')) == False:
        os.makedirs(os.path.join(save_path,dir,'mask'))

    image_path = os.path.join(input_path,dir,'image')
    mask_path = os.path.join(input_path,dir,'mask')
    for file in os.listdir(image_path):
        img = sitk.ReadImage(os.path.join(image_path, file))
        img = sitk.GetArrayFromImage(img)
        img = 255 - img * 255
        img = img.astype(np.uint8)



        euclidean_distance = np.linalg.norm(target_pixel - img, axis=2)
        min_euclidean_distance_loc = np.min(euclidean_distance)

        if min_euclidean_distance_loc < distance_threshold:
            similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)
            similarity_spot_num = similarity_spot_loc.shape[0]
            if similarity_spot_num < 100:
                continue

            mask = sitk.ReadImage(os.path.join(mask_path, file))
            mask = sitk.GetArrayFromImage(mask)[:, :, 0]

            mask_ul = mask[:100, :100]
            similarity_spot_loc = np.argwhere(mask_ul == 1)
            ul_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_ur = mask[:50, -50:]
            similarity_spot_loc = np.argwhere(mask_ur == 1)
            ur_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_dl = mask[-50:, :50]
            similarity_spot_loc = np.argwhere(mask_dl == 1)
            dl_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_dr = mask[-100:, -100:]
            similarity_spot_loc = np.argwhere(mask_dr == 1)
            dr_similarity_spot_num = similarity_spot_loc.shape[0]

            if dl_similarity_spot_num < 50 or ur_similarity_spot_num < 50:
                continue

            if dr_similarity_spot_num > 0 or ul_similarity_spot_num > 0:
                continue

            img_ul = img[:100, :100]
            img_dr = img[-100:, -100:]
            if np.mean(img_ul) > 200:
                continue

            if np.mean(img_ul) < 150 and np.mean(img_dr) < 150:
                continue
            plt.imsave(os.path.join(save_path, dir, 'jpg', file.replace('nii', 'jpg')), img)

            mask = sitk.GetImageFromArray(mask)
            sitk.WriteImage(mask, os.path.join(save_path,dir,'mask',file))

            img = sitk.GetImageFromArray(img)
            sitk.WriteImage(img, os.path.join(save_path,dir,'image',file))

