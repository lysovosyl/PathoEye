#%%
import csv
import os
import torch
import torch.nn as nn
import numpy as np
from models.patheye import GradCamModel
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import matplotlib as mpl
#%%
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-test_path',type=str,required=True,help='this directory should be generator from create_patches.py')
parse.add_argument('-model_path',type=str,required=True,help='the directory should be generated from train')
parse.add_argument('-save_path',type=str,required=True,help='this directory will save the config of the model and its weight')
parse.add_argument('-device',type=str,default='cuda:0',help='The device on which to run this program. The default is cuda:0.')
args = parse.parse_args()

test_path = args.test_path
model_path = args.model_path
save_path = args.save_path
device = args.device

#%%
with open(os.path.join(model_path,'config.json'), 'r', encoding='utf-8') as f:
    model_config = json.load(f)
class_num = model_config['class_num']
model = GradCamModel(class_num).to(device)
model.load_state_dict(torch.load(model_config['model'],weights_only=True))
#%%

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


data = {}
for index,disease in enumerate(os.listdir(test_path)):
    data[disease]= {}
    for patient in os.listdir(os.path.join(test_path, disease)):
        data[disease][patient] = []
        for img in os.listdir(os.path.join(test_path, disease, patient, 'data')):
            data[disease][patient].append(os.path.join(test_path, disease, patient, 'data', img))



#%%
explain_path = os.path.join(save_path,'explain_img')
if os.path.exists(explain_path) == False:
    os.makedirs(explain_path)
predict_path = os.path.join(save_path,'predict')
if os.path.exists(predict_path) == False:
    os.makedirs(predict_path)

#%%
from torch.utils.data import dataset
def class_index(path):
    label_index = {}
    for index, disease in enumerate(os.listdir(path)):
        label_index[disease] = index
    return label_index
class LoadData(dataset.Dataset):
    def __init__(self,path,class_num):
        super(LoadData, self).__init__()
        self.data = {}
        self.disease_label = {}
        label_index = class_index(path)
        i = 0
        for index,disease in enumerate(os.listdir(path)):
            self.disease_label[disease] = np.zeros([class_num])
            self.disease_label[disease][label_index[disease]] = 1
            self.data[i] = {}
            for patient in os.listdir(os.path.join(path, disease)):
                for img in os.listdir(os.path.join(path, disease, patient, 'data')):
                    self.data[i] = {
                        'disease':disease,
                        'patient':patient,
                        'path':os.path.join(path, disease, patient, 'data', img)
                    }
                    i+=1


        self.data_len = len(self.data)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])


    def __getitem__(self,index):

        path = self.data[index]['path']
        disease = self.data[index]['disease']
        patient = self.data[index]['patient']
        img = plt.imread(path)
        img = np.array(img[:,:,:3])
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.disease_label[disease]
        label = torch.from_numpy(label)

        self.success_index = index
        return img, label,patient,disease,path
    def __len__(self):
        return self.data_len
#%%
result = {}
for disease in data:
    result[disease] = {}
    for patient in data[disease].keys():
        result[disease][patient] = {}
        result[disease][patient]['img'] = []
        result[disease][patient]['prob'] = []
        result[disease][patient]['pred'] = []
        result[disease][patient]['true'] = []
model.eval()
from torch.utils.data import dataloader
from tqdm import tqdm
test_data = LoadData(test_path, class_num)
test_dataloader = dataloader.DataLoader(test_data, batch_size=10,shuffle=True,num_workers=32)
progress_bar = tqdm(test_dataloader, desc=f"Test: ")
for img,label,patient_list,disease_list,path_list in progress_bar:
    img = img.float()
    img = img.to(device)
    outputs_list, acts_list = model(img)
    for index in range(len(outputs_list)):
        patient = patient_list[index]
        disease = disease_list[index]
        path    = path_list[index]
        outputs = outputs_list[index:index+1,:]
        acts   =  acts_list[index:index+1,:]
        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.cpu().detach().numpy()
        predict = np.argmax(outputs)
        result[disease][patient]['img'].append(path)
        result[disease][patient]['prob'].append(outputs)
        result[disease][patient]['pred'].append(predict)
        result[disease][patient]['true'].append(model_config['label'][disease])
#%%
f = open(os.path.join(predict_path,'predict_info.txt'),'w')
writer = csv.writer(f)
for disease in result:
    for patient in result[disease]:
        for index in range(len(result[disease][patient]['img'])):
            writer.writerow([result[disease][patient]['img'][index],result[disease][patient]['prob'][index],result[disease][patient]['pred'][index],result[disease][patient]['true'][index]])
f.close()

#%%
print('True','Predict')
for disease in result:
    for patient in result[disease]:
        num = []
        for i in range(model_config['class_num']):
            num.append(result[disease][patient]['pred'].count(i))
        print(disease,patient,list(model_config['label'].keys())[np.argmax(num)],num)