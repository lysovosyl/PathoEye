#%%
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



target_pixel = np.array([[134.71875,52.84375,99.375]])
# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/cut_image/128'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_128'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True,help='the dir of segmentation image(128*128) result from 01_cut_image.py')
parser.add_argument('-save_path',type=str,required=True,help='the dir of save filter result(dir path)')
args = parser.parse_args()

input_path = args.input_path
save_path = args.save_path

distance_threshold = 20

for index,dir in enumerate(os.listdir(input_path)):
    if index < 495 or index >500:
        continue
    print(index,len(os.listdir(input_path)))
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

            mask_ul = mask[:50, :50]
            similarity_spot_loc = np.argwhere(mask_ul == 1)
            ul_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_ur = mask[:32, -32:]
            similarity_spot_loc = np.argwhere(mask_ur == 1)
            ur_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_dl = mask[-32:, :32]
            similarity_spot_loc = np.argwhere(mask_dl == 1)
            dl_similarity_spot_num = similarity_spot_loc.shape[0]

            mask_dr = mask[-32:, -32:]
            similarity_spot_loc = np.argwhere(mask_dr == 1)
            dr_similarity_spot_num = similarity_spot_loc.shape[0]

            if dr_similarity_spot_num < 300:
                continue
            if ul_similarity_spot_num > 200:
                continue
            if ur_similarity_spot_num > 400:
                continue
            if dl_similarity_spot_num > 400:
                continue

            plt.imsave(os.path.join(save_path, dir, 'jpg', file.replace('nii', 'jpg')), img)

            mask = sitk.GetImageFromArray(mask)
            sitk.WriteImage(mask, os.path.join(save_path,dir,'mask',file))

            img = sitk.GetImageFromArray(img)
            sitk.WriteImage(img, os.path.join(save_path,dir,'image',file))

