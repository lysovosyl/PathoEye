#%%
import sys
sys.path.append('../')

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from skimage import segmentation
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from models.InfoSeg import InfoSeg



class method_infoseg():
    def __init__(self,input_channel=3):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.label_colours = np.random.randint(255, size=(100, 3))  # 随机生成像素颜色。100对应100层特征图
        self.im_shape = []
        self.nChannel = 24
        self.nConv = 2
        self.model = InfoSeg(input_channel, nChannel=self.nChannel, nConv=self.nConv)
        self.img = None
        self.transform = transforms.ToTensor()
        self.maxIter = 100
        self.lr = 0.1
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.l_inds = []

    def reset_model(self,input_channel=3):
        self.model = InfoSeg(input_channel, nChannel=self.nChannel, nConv=self.nConv)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def filter_mask_select_pixel(self,img, mask, pixel=[174.04746886, 136.90078296, 193.84494564]):
        label_distance = {}
        for index, label in enumerate(np.unique(mask)):
            pixel_list = img[mask == label]
            target_pixel = np.array(pixel)
            region_pexel = np.mean(pixel_list, axis=0)
            euclidean_distance = np.linalg.norm(target_pixel - region_pexel)
            label_distance[label] = euclidean_distance
        select_label = min(label_distance, key=label_distance.get)

        mask_eb = np.zeros_like(mask).astype(np.uint8)
        mask_eb[mask == select_label] = 1
        # 定义结构元素 (核) 5x5 的矩形核，你可以根据需求调整大小
        kernel = np.ones((5, 5), np.uint8)
        mask_eb_eroded = cv2.erode(mask_eb, kernel, iterations=1)
        mask_eb_dilated = cv2.dilate(mask_eb_eroded, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask_eb_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 5000:
                mask_eb = cv2.drawContours(mask_eb_dilated, [contours[i]], -1, 0, thickness=-1)
        return mask_eb

    def seg_image(self,img,device='cuda:0',minLabels=4,target_pixel = [174.04746886, 136.90078296, 193.84494564]):
        labels = segmentation.felzenszwalb(img, scale=int(64), sigma=0.5,min_size=int(128))
        labels = labels.reshape(img.shape[0] * img.shape[1])
        u_labels = np.unique(labels)
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

        img = torch.from_numpy(img).to(device).type(torch.float)
        img = img.permute([2,0,1])
        img = img.unsqueeze(dim=0)
        self.reset_model()
        self.model = self.model.to(device)
        for batch_idx in range(self.maxIter):
            self.optimizer.zero_grad()
            output = self.model(img)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
            ignore, target = torch.max(output, 1)
            self.mask = target.data.cpu().numpy()
            for i in range(len(self.l_inds)):
                labels_per_sp = self.mask[self.l_inds[i]]  # im_target指的是图像拉成一维数组后对应像素的颜色标签
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                self.mask[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(self.mask)
            target = target.to(device)
            target = Variable(target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            nLabels = len(np.unique(self.mask))
            if nLabels <= minLabels:
                break

        img = img.squeeze()
        img = img.permute([1, 2, 0])
        img = img.cpu().numpy().astype(np.uint8)
        mask_all = self.mask.reshape(img.shape[:2])
        mask_target = self.filter_mask_select_pixel(img, mask_all)

        del output, loss, target,ignore
        torch.cuda.empty_cache()

        return img,mask_target

#%%
