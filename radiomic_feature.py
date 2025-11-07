import os
from radiomics import featureextractor
import six
import csv

#%%
def draw_feature(image_path,mask_path,save_path):
    config_path = './original.yaml'
    feature = {}
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)

    result = extractor.execute(image_path, mask_path)

    for key, val in (six.iteritems(result)):
        try:
            val = float(val)
        except:
            continue
        feature[key] = val

    f = open(save_path, 'w')
    writer = csv.writer(f)
    for feature_key in feature.keys():
        writer.writerow([feature_key, feature[feature_key]])
    f.close()

def run_mission(mission:dict):
    for key in mission.keys():
        image_path = mission[key]['image_path']
        mask_path  = mission[key]['mask_path']
        save_path  = mission[key]['out_path']
        try:
            draw_feature(image_path,mask_path,save_path)
        except:
            continue

#%%
# lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/lable_train'
# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_128/'
# save_path  = 'cd '
# thread = 70
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-lable_path',type=str,required=True)
parse.add_argument('-input_path',type=str,required=True)
parse.add_argument('-save_path',type=str,required=True)
parse.add_argument('-thread',type=int,default=30)
args = parse.parse_args()
lable_path = args.lable_path
input_path = args.input_path
save_path  = args.save_path
thread = args.thread
f = open(lable_path)
lines = f.readlines()
file_type = {}
for line in lines:
    line = line[:-1].split('\t')
    file = '-'.join(line[0].split('-')[0:3])
    type = line[1]
    if type not in file_type.keys():
        file_type[type] = []
    file_type[type].append(file)
assert len(file_type.keys()) == 2

#%%

path_dict = {}
i = 0

for type in file_type.keys():
    for dir in file_type[type]:
        if dir in os.listdir(os.path.join(input_path)):
            for file in os.listdir(os.path.join(input_path,dir,'image')):
                image_path = os.path.join(input_path,dir,'image',file)
                mask_path = os.path.join(input_path,dir,'mask',file)
                out_path  = os.path.join(save_path,dir,file.replace('nii','csv'))
                path_dict['{}_{}_{}'.format(type,dir,i)] = {
                    'image_path':image_path,
                    'mask_path':mask_path,
                    'out_path':out_path,
                    'name':'{}_{}_{}'.format(type,dir,i)
                }
                if os.path.exists(os.path.join(save_path,dir)) == False:
                    os.makedirs(os.path.join(save_path,dir))
                i+=1
#%%


def split_list(input_list, num_parts):
    avg = len(input_list) / num_parts
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg

    return out

result = split_list(list(path_dict.keys()), thread)

mission = {}
for i,mission_list in enumerate(result):
    mission[i] = {}
    for j in mission_list:
        mission[i][j] = path_dict[j]

#%%
import multiprocessing



jobs = []
for i in mission.keys():
    p = multiprocessing.Process(target=run_mission, args=(mission[i],))
    jobs.append(p)
    p.start()


complete_num = 0
import time
while True:
    for p in jobs:
        if p.is_alive():
            print('\r',"Child process is still running...",end=' ')
            time.sleep(5)
            break
        else:
            complete_num+=1

    if complete_num==thread:
        break