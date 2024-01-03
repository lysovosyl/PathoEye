#%%
import os
import random
import shutil
from sklearn.model_selection import train_test_split

data_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/image_512'
lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/lable/hist.txt'
save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/deeplearning_image_512'

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-input_path',type=str,required=True,help='the dir of filted image,which result from 02_select_128(512)_image.py')
# parser.add_argument('-label_path',type=str,required=True,help='the label describe data age')
# parser.add_argument('-save_path',type=str,required=True,help='the dir of save dataset')
# args = parser.parse_args()
#
# data_path = args.input_path
# lable_path = args.label_path
# save_path = args.save_path


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

for i in chose_age:
    if os.path.exists(os.path.join(save_path,'train',str(i))) == False:
        os.makedirs(os.path.join(save_path,'train',str(i)))
for i in chose_age:
    if os.path.exists(os.path.join(save_path,'test',str(i))) == False:
        os.makedirs(os.path.join(save_path,'test',str(i)))

for i,dir in enumerate(os.listdir(data_path)):
    if tab_dict[dir] in chose_age:
        for file in os.listdir(os.path.join(data_path,dir,'bmp')):
            age_image[tab_dict[dir]].append(os.path.join(data_path,dir,'bmp',file))

data_len = []
for i in age_image.keys():
    data_len.append(len(age_image[i]))
min_data_len = min(data_len)
for i in age_image.keys():
    age_image[i] = random.sample(age_image[i],min_data_len)

x = []
y = []
for i in age_image.keys():
    for j in age_image[i]:
        x.append(j)
        y.append(i)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
for index,(i,j) in enumerate(zip(x_train,y_train)):
    source_path = i
    target_path = os.path.join(save_path,'train',str(j),str(index)+'.bmp')
    shutil.copy(source_path,target_path)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
for index,(i,j) in enumerate(zip(x_test,y_test)):
    source_path = i
    target_path = os.path.join(save_path,'test',str(j),str(index)+'.bmp')
    shutil.copy(source_path,target_path)
