from torch.utils.data import dataset
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#%%
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
        num = []
        label_index = class_index(path)
        for index,disease in enumerate(os.listdir(path)):
            self.disease_label[disease] = np.zeros([class_num])
            self.disease_label[disease][label_index[disease]] = 1
            self.data[disease] = []
            for patient in os.listdir(os.path.join(path, disease)):
                for img in os.listdir(os.path.join(path, disease, patient, 'data')):
                    self.data[disease].append(os.path.join(path, disease, patient, 'data', img))
            num.append(len(self.data[disease]))
        self.data_len = min(num)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)

        ])


    def __getitem__(self,index):
        img_list = []
        label_list = []
        try:
            for disease in self.data:
                img = plt.imread(self.data[disease][index])
                img = np.array(img[:,:,:3])
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transform(img)
                label = self.disease_label[disease]
                label = torch.from_numpy(label)
                img_list.append(img)
                label_list.append(label)
                self.success_index = index
        except:
            index = self.success_index
            for disease in self.data:
                img = np.zeros([512,512,3],dtype=np.uint8)
                img = self.transform(img)
                label = self.disease_label[disease]
                label = torch.from_numpy(label)
                img_list.append(img)
                label_list.append(label)
        return img_list, label_list
    def __len__(self):
        return self.data_len