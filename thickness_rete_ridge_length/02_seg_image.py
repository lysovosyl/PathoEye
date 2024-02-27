#from __future__ import print_function
#%%
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from skimage import morphology

#%%
# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim,nChannel,nConv):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)
        self.nConv = nConv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
#%%
class seg_model():
    nChannel = 100
    maxIter = 400
    minLabels = 4
    lr = 0.1
    nConv = 2
    num_superpixels = 10000
    compactness = 100
    visualize = 1
    l_inds = []
    def __init__(self,cuda,visualize=1):

        self.cuda = cuda

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.label_colours = np.random.randint(255, size=(100, 3))  # 随机生成像素颜色。100对应100层特征图
        self.visualize = visualize
        self.im_shape = []

    def build_model(self):
        self.model = MyNet(self.data.size(1),nChannel=self.nChannel,nConv=self.nConv)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.model = self.model.to(self.cuda)
        self.model.train()

    def draw_model(self):
        with SummaryWriter(comment="FPN") as w:
            w.add_graph(model=self.model, input_to_model=torch.rand(15, 3, 224, 224))

    def Model_segmentation(self):
        for batch_idx in range(self.maxIter):
            self.optimizer.zero_grad()
            output = self.model(self.data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)  # permute将tensor的维度换位 contiguous()修改底层数据 view按照指定二维展开
            ignore, target = torch.max(output, 1)  # 取数值最大者为对应像素的标签 返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
            self.im_target = target.data.cpu().numpy()  # 先转换为cpu tensor在转换为numpy格式数据，不能直接由cuda转numpy
            nLabels = len(np.unique(self.im_target))
            if self.visualize:
                im_target_rgb = np.array([self.label_colours[c % 100] for c in self.im_target])
                im_target_rgb = im_target_rgb.reshape(self.im.shape).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)
            # superpixel refinement
            # TODO: use Torch Variable instead of numpy for faster calculation
            for i in range(len(self.l_inds)):
                labels_per_sp = self.im_target[self.l_inds[i]]  # im_target指的是图像拉成一维数组后对应像素的颜色标签
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                self.im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(self.im_target)
            target = target.to(self.cuda)
            target = Variable(target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if nLabels <= self.minLabels:
                break
            # save output image
            if not self.visualize:
                output = self.model(self.data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
                ignore, target = torch.max(output, 1)
                self.im_target = target.data.cpu().numpy()
                im_target_rgb = np.array([self.label_colours[c % 100] for c in self.im_target])
                im_target_rgb = im_target_rgb.reshape(self.im.shape).astype(np.uint8)
        # cv2.imwrite("output.bmp", im_target_rgb)

    def Post_segmentation_processing(self):
        im_2 = self.im_target.reshape(self.im.shape[0:2]).astype(np.uint8)
        u_labels = np.unique(self.im_target)  # 获取分割后图像的标签类型
        Randomly_collect_the_number_of_pixels = 50
        Image_to_be_processed = np.array(self.im)
        Image_to_be_processed = Image_to_be_processed.reshape(-1, 3)
        Randomly_extract_pixel_mean = []
        for labels in u_labels:
            pixels_loaction = np.where(self.im_target == labels)[0]
            pixels_loaction = pixels_loaction[np.random.randint(0, len(pixels_loaction), Randomly_collect_the_number_of_pixels)]
            Randomly_extract_pixels = Image_to_be_processed[pixels_loaction]
            Randomly_extract_pixel_mean.append(np.mean(Randomly_extract_pixels, axis=0))
        Target_pixel_color = np.array([[77.48, 36.72, 107.56]])
        Target_pixel_color = Target_pixel_color.repeat(repeats=len(u_labels), axis=0)
        Randomly_extract_pixel_mean = np.array(Randomly_extract_pixel_mean)
        Euclidean_distance = np.linalg.norm(Target_pixel_color - Randomly_extract_pixel_mean, axis=1)
        Minimum_Euclidean_distance_index = Euclidean_distance.argmin()
        # if Minimum_Euclidean_distance_index < 30:
        #     print('找到目标像素')
        im_2[im_2 != u_labels[Minimum_Euclidean_distance_index]] = 0
        im_2[im_2 == u_labels[Minimum_Euclidean_distance_index]] = 255
        kernel1 = np.ones((3, 3), np.uint8)
        im_2 = cv2.dilate(im_2, kernel1)
        im_2 = cv2.erode(im_2, kernel1)

        im_2 = cv2.erode(im_2, kernel1)
        im_2 = cv2.dilate(im_2, kernel1)
        contours, hierarchy = cv2.findContours(im_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 5000:
                im_2 = cv2.drawContours(im_2, [contours[i]], -1, 0, thickness=-1)
        im_3 = 255 * np.ones(self.im.shape[0:2]).astype(np.uint8)
        im_2 = np.array(im_2).astype(np.uint8)

        im_3 = np.subtract(im_3, im_2).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 500:
                im_3 = cv2.drawContours(im_3, [contours[i]], -1, 0, thickness=-1)
        im_2 = 255 * np.ones(self.im.shape[0:2]).astype(np.uint8)
        im_2 = np.subtract(im_2, im_3).astype(np.uint8)
        self.im_2 = im_2
        self.im_3 = im_3

    def load_pic(self,path):
        self.im = cv2.imread(path)
        self.data = torch.from_numpy(
            np.array([self.im.transpose((2, 0, 1)).astype('float32') / 255.]))  # 从numpy.ndarray创建一个张量 transpose表示调换元素位置
        self.data = self.data.to(self.cuda)

    def Pathological_tissue_segmentation(self):

        labels = segmentation.felzenszwalb(self.im, scale=32, sigma=0.5, min_size=64)
        labels = labels.reshape(self.im.shape[0] * self.im.shape[1])
        u_labels = np.unique(labels)
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])
        self.build_model()
        self.Model_segmentation()
        self.Post_segmentation_processing()
        return self.im_2,self.im_3
#measured_thickness
#im_2 二值图像 目标区域为255
#im_3 二值图像 目标区域为0
def measured_thickness(im_2, im_3):
    Surround_Area, hierarchy = cv2.findContours(im_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Area = cv2.contourArea(Surround_Area[0])
    contours, hierarchy = cv2.findContours(im_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_3, contours, -1, color=(0, 255, 255))

    contours_del = []
    contours_len = []
    for i in range(len(contours)):
        temp = np.array(contours[i]).reshape(-1, 2)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 1] < 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)

        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 0] < 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)

        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 1] > im_2.shape[0] - 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)

        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 0] > im_2.shape[1] - 10:
                delate_loc.append(j)
        contours_del.append(temp)
        contours_len.append(temp.shape[0])

    def top_k(num_list, n):
        '''求list前n个最大值'''
        n %= len(num_list)
        pad = min(num_list) - 1  # 最小值填充
        topn_list = []
        for i in range(n):
            topn_list.append(max(num_list))
            max_idx = num_list.index(max(num_list))  # 找最大值索引
            num_list[max_idx] = pad  # 最大值填充
        return topn_list

    # 线性回归
    contours1 = []
    temp = contours_len.copy()
    if len(contours_len) == 2:
        list_top2 = contours_len
    else:
        list_top2 = top_k(temp, 2)

    for i in list_top2:
        contours1.append(contours_del[contours_len.index(i)])
    distence = 0
    for i in contours1:
        x = np.array(i[:, 0]).reshape(-1)
        y = np.array(i[:, 1]).reshape(-1)
        model = np.polyfit(x, y, deg=1)
        xpredict0 = np.min(x)
        xpredict1 = np.max(x)
        ypredict0 = np.polyval(model, np.min(x))
        ypredict1 = np.polyval(model, np.max(x))
        distence += np.linalg.norm(np.array(xpredict0, ypredict0) - np.array(xpredict1, ypredict1))
    length = distence / 2
    thickness = Area / length
    return thickness
#%%
#im_2 二值图像 目标区域为255
#im_3 二值图像 目标区域为0
def Thickness_of_each_point(im_2,im_3):
    contours, hierarchy = cv2.findContours(im_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_del = []
    contours_len = []
    #这里寻找出边缘，除去图像的边缘
    for i in range(len(contours)):
        temp = np.array(contours[i]).reshape(-1, 2)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 1] < 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 0] < 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 1] > im_2.shape[0] - 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 0] > im_2.shape[1] - 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        contours_del.append(temp)
        contours_len.append(temp.shape[0])

    def top_k(num_list, n):
        '''求list前n个最大值'''
        n %= len(num_list)
        pad = min(num_list) - 1  # 最小值填充
        topn_list = []
        for i in range(n):
            topn_list.append(max(num_list))
            max_idx = num_list.index(max(num_list))  # 找最大值索引
            num_list[max_idx] = pad  # 最大值填充
        return topn_list

    Organizational_edge = []
    temp = contours_len.copy()
    if len(contours_len) <2:
        print('异常数据')
        return -1,-1
    elif len(contours_len) == 2:
        list_top2 = contours_len
    else:
        list_top2 = top_k(temp, 2)
    for i in list_top2:
        Organizational_edge.append(contours_del[contours_len.index(i)])

    Euclidean_distance = [[],[]]
    for i in range(Organizational_edge[0].shape[0]):
        Coordinate_1 = Organizational_edge[0][i]
        Coordinate_1 = np.full(Organizational_edge[1].shape,Coordinate_1)
        temp = np.linalg.norm(Coordinate_1 - Organizational_edge[1],axis=1)
        Euclidean_distance[0].append(np.min(temp))

    for i in range(Organizational_edge[1].shape[0]):
        Coordinate_1 = Organizational_edge[1][i]
        Coordinate_1 = np.full(Organizational_edge[0].shape,Coordinate_1)
        temp = np.linalg.norm(Coordinate_1 - Organizational_edge[0],axis=1)
        Euclidean_distance[1].append(np.min(temp))

    return Euclidean_distance  #一般来说0为下一点的边缘，1为上一点的边缘
#%%

import sys
import csv
import os
import shutil


input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/cut_image_for_fold_and_thickness/'
save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/fold_and_thickness_seg_data/'
cuda = 'cuda:0'

for patient in os.listdir(input_path):
    for img_file in os.listdir(os.path.join(input_path,patient)):
        input_img = os.path.join(input_path,patient,img_file)
        save_dir = os.path.join(save_path,patient,img_file.split('.')[0])

        Organization_1 = seg_model(cuda=cuda,visualize=0)
        Organization_1.load_pic(input_img)
        Target_light, Target_Dark= Organization_1.Pathological_tissue_segmentation()
        im4 = Target_light[:,:,np.newaxis]
        im4 = np.repeat(im4,repeats=3,axis=2)
        plt.imshow(im4)
        plt.show()
        a,b = Thickness_of_each_point(Target_light, Target_Dark)


        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        print(input_img,save_dir)
        if len(a) > 2 and len(b) > 2:
            f = open(os.path.join(save_dir,'thickness_a.csv'),'w')
            writer = csv.writer(f)
            writer.writerow(a)
            f.close()
            f = open(os.path.join(save_dir,'thickness_b.csv'),'w')
            writer = csv.writer(f)
            writer.writerow(b)
            f.close()
            np.save(os.path.join(save_dir,'basecell_light_255.npy'),Target_light)
            np.save(os.path.join(save_dir,'basecell_dark_000.npy'),Target_Dark)
            shutil.copy(input_img,os.path.join(save_dir,'raw_image.bmp'))