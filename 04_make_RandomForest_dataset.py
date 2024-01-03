import os
import pandas
import csv
import random
import numpy as np

# data_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/pyradiomic_512'
# lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/lable/hist.txt'
# save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test'
# save_file_name = 'default_name.csv'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True,help='the dir of filted image,which result from 02_select_128(512)_image.py')
parser.add_argument('-label_path',type=str,required=True,help='the label describe data age')
parser.add_argument('-save_path',type=str,required=True,help='the dir of save dataset')
parser.add_argument('-save_file_name',type=str,default='train_features.csv',help='the dir of save dataset')
args = parser.parse_args()

data_path = args.input_path
lable_path = args.label_path
save_path = args.save_path
save_file_name = args.save_file_name

age_reflect = {
'20-29':20,
'30-39':30,
'40-49':40,
'50-59':50,
'60-69':60,
'70-79':70,
}

tab_file = open(lable_path)
lines = tab_file.readlines()
tab_dict = {}
for line in lines:
    line = line.split('\t')
    file_name = line[0].split('-')
    file_name = file_name[0] + '-' + file_name[1] + '-' + file_name[2]
    tab_dict[file_name] = age_reflect[line[2]]
chose_age = [20,70]
age_image = {}
for i in chose_age:
    age_image[i] = []



#%%
if os.path.exists(save_path) == False:
    os.mkdir(save_path)

data1 = {}
for index,dir in enumerate(os.listdir(data_path)):
    if tab_dict[dir] in chose_age:
        for loc,file in enumerate(os.listdir(os.path.join(data_path,dir))):
            data1['{}_{}_{}'.format(dir,loc,tab_dict[dir])] = {}
            reader = csv.reader(open(os.path.join(data_path,dir,file)))
            for name,value in reader:
                data1['{}_{}_{}'.format(dir,loc,tab_dict[dir])][name] = float(value)
df = pandas.DataFrame(data1)
df = df[df.var(axis=1) != 0]
df_normalized = df.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)
df_normalized = df_normalized.dropna(axis=0, how='any')
df_normalized = df_normalized.dropna(axis=1, how='any')

#%%

columns = df_normalized.columns
type_count = {}
type_name = {}
for i in chose_age:
    type_count[str(i)] = 0
    type_name[str(i)] = []

for i in columns:
    type = i.split('_')[-1]
    type_count[type] += 1
    type_name[type].append(i)

num = min(type_count.values())


header = []
for type in chose_age:
    header.extend(random.sample(type_name[str(type)],num))
new_df = df[header]
new_df.to_csv(os.path.join(save_path,save_file_name))
