import os
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import six
import csv
import sys

# dir = sys.argv[1]
# input_path = sys.argv[2]
# save_path = sys.argv[3]


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True,help='the dir of filted image,which result from 02_select_128(512)_image.py')
parser.add_argument('-save_path',type=str,required=True,help='the dir of save extract features result(dir path)')
parser.add_argument('-config_path',type=str,required=True,help='chose from config dir,example: /your_path/original.yaml')
parser.add_argument('-mask_path',type=str,required=True,help='chose from mask dir,you has two choice(128*128,512*512),it depend on what size of you image')
args = parser.parse_args()


# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_128'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/github_test_feature'
# config_path = '/mnt/dfc_data2/project/linyusen/project/tmp/pycharm_project_129/config/original.yaml'
# mask_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/mask_128.nii'

input_path = args.input_path
save_path = args.save_path
config_path = args.config_path
mask_path = args.mask_path

if os.path.exists(save_path) == False:
    os.makedirs(save_path)
#%%
img_path_list = {}
for dir in os.listdir(input_path):
    for img_file in os.listdir(os.path.join(input_path,dir,'image')):
        if dir not in img_path_list.keys():
            img_path_list[dir] = []
        img_path_list[dir].append(os.path.join(input_path,dir,'image',img_file))




#%%
for dir in img_path_list.keys():
    if os.path.exists(os.path.join(save_path,dir)) == False:
        os.makedirs(os.path.join(save_path,dir))
    for index in range(len(img_path_list[dir])):
        feature = {}
        extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
        img_path = os.path.join(input_path,img_path_list[dir][index])
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        if np.max(mask) == 0:
            continue
        if np.min(mask) == 1:
            continue
        result = extractor.execute(img_path, mask_path)

        for key, val in (six.iteritems(result)):
            try:
                val = float(val)
            except:
                continue
            feature[key] = val


        f = open(os.path.join(save_path,dir,os.path.split(img_path)[-1].split('.')[0]+'_'+str(index)+'.csv'), 'w')
        writer = csv.writer(f)
        for feature_key in feature.keys():
            writer.writerow([feature_key, feature[feature_key]])
        f.close()
