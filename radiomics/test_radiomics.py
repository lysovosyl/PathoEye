import csv
import numpy as np
import os
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('-input_path',type=str)
parse.add_argument('-model_path',type=str)
parse.add_argument('-mean_path',type=str)
parse.add_argument('-std_path',type=str)
args = parse.parse_args()

input_path = args.input_path
model_path = args.model_path
mean_path = args.mean_path
std_path = args.std_path


import pickle
with open(mean_path, 'rb') as file:
    mean = pickle.load(file)

with open(std_path, 'rb') as file:
    std = pickle.load(file)

#%%
feature = {}
for file_name in os.listdir(input_path):
    feature[file_name] = {}
    for feature_file in os.listdir(os.path.join(input_path,file_name)):
        feature[file_name][feature_file] = {}
        f = open(os.path.join(input_path,file_name,feature_file))
        reader = csv.reader(f)
        i=0
        for line in reader:
            name = line[0]
            if name in ['diagnostics_Mask-original_VolumeNum','diagnostics_Mask-interpolated_VolumeNum']:
                continue
            value= (float(line[1])-mean[i])/std[i]
            feature[file_name][feature_file][name] = value
            i+=1

#%%

model = pickle.load(open(model_path, 'rb'))
result = {}
for patient in feature.keys():
    result[patient] = []
    for feature_file in feature[patient].keys():
        test_x = feature[patient][feature_file].values()
        test_x = list(test_x)
        test_x = np.array(test_x)
        test_x = test_x.reshape(1, -1)
        predict_y = model.predict(test_x)
        result[patient].append(predict_y)

#%%
lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/lable_train'
f = open(lable_path)
lines = f.readlines()
file_type = {}
for line in lines:
    line = line[:-1].split('\t')
    file = '-'.join(line[0].split('-')[0:3])
    type = line[1]
    file_type[file]=type

for patient in result.keys():
    if np.average(result[patient]) < 0.5:
        print('predict old','true ',file_type[patient])
    else:
        print('predict young','true ',file_type[patient])
