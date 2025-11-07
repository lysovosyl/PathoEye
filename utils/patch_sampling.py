import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
target_pixel = np.array([[134.71875,52.84375,99.375]])
# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/cut_image/128'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_128'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True)
parser.add_argument('-save_path',type=str,required=True)
parser.add_argument('-p_start',type=int,default=0)
parser.add_argument('-p_end',type=int,default=99999999)
args = parser.parse_args()

input_path = args.input_path
save_path  = args.save_path
p_start = args.p_start
p_end = args.p_end

#%%
distance_threshold = 20
all_img = 0
for index,dir in enumerate(os.listdir(input_path)):
    print('\r', index, '/', len(os.listdir(input_path)), end=' ')
    if index > p_start and index < p_end:

        if os.path.exists(os.path.join(save_path,dir)) == False:
            os.makedirs(os.path.join(save_path,dir))

        if os.path.exists(os.path.join(save_path,dir,'bmp')) == False:
            os.makedirs(os.path.join(save_path,dir,'bmp'))
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


            blank_pixel = np.array([255,255,255])
            euclidean_distance = np.linalg.norm(blank_pixel - img, axis=2)
            similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)
            similarity_spot_num = similarity_spot_loc.shape[0]
            if similarity_spot_num > 300:
                continue


            select_ul = []
            select_dr = []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if j < img.shape[1] - i:
                        select_ul.append(img[i, j, :])
                    else:
                        select_dr.append(img[i, j, :])
            select_ul = np.array(select_ul)
            select_dr = np.array(select_dr)

            euclidean_distance = np.linalg.norm(target_pixel - select_ul, axis=1)
            similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)
            similarity_spot_num_ul = similarity_spot_loc.shape[0]

            euclidean_distance = np.linalg.norm(target_pixel - select_dr, axis=1)
            similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)
            similarity_spot_num_dr = similarity_spot_loc.shape[0]

            if similarity_spot_num_dr > 1000 and similarity_spot_num_ul < 50:
                plt.imsave(os.path.join(save_path, dir, 'bmp', file.replace('nii', 'bmp')), img)

                mask = sitk.ReadImage(os.path.join(mask_path, file))
                mask = sitk.GetArrayFromImage(mask)
                mask[1:127,1:127,:] = 1



                mask = sitk.GetImageFromArray(mask)
                sitk.WriteImage(mask, os.path.join(save_path, dir, 'mask', file))

                img = sitk.ReadImage(os.path.join(image_path, file))
                img = sitk.GetArrayFromImage(img)

                img = sitk.GetImageFromArray(img)
                sitk.WriteImage(img, os.path.join(save_path, dir, 'image', file))