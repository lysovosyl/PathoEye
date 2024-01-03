#%%
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init
from torch.utils.tensorboard import SummaryWriter
import openslide
import os
import argparse


class InfoSeg(nn.Module):
    def __init__(self,input_dim,nChannel,nConv):
        super(InfoSeg, self).__init__()
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
class Seg_image():
    nChannel = 100
    maxIter = 400
    minLabels = 4
    lr = 0.1
    nConv = 2
    num_superpixels = 10000
    compactness = 100
    l_inds = []
    def __init__(self):

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.label_colours = np.random.randint(255, size=(100, 3))  # 随机生成像素颜色。100对应100层特征图
        self.im_shape = []

    def build_model(self,device):
        self.model = InfoSeg(self.data.size(1),nChannel=self.nChannel,nConv=self.nConv)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.model = self.model.to(device)
        self.model.train()

    def draw_model(self):
        with SummaryWriter(comment="FPN") as w:
            w.add_graph(model=self.model, input_to_model=torch.rand(15, 3, 224, 224))

    def pre_seg(self,device):
        for batch_idx in range(self.maxIter):
            self.optimizer.zero_grad()
            output = self.model(self.data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)  # permute将tensor的维度换位 contiguous()修改底层数据 view按照指定二维展开
            ignore, target = torch.max(output, 1)  # 取数值最大者为对应像素的标签 返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
            self.im_target = target.data.cpu().numpy()  # 先转换为cpu tensor在转换为numpy格式数据，不能直接由cuda转numpy
            nLabels = len(np.unique(self.im_target))
            for i in range(len(self.l_inds)):
                labels_per_sp = self.im_target[self.l_inds[i]]  # im_target指的是图像拉成一维数组后对应像素的颜色标签
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                self.im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(self.im_target)
            target = target.to(device)
            target = Variable(target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if nLabels <= self.minLabels:
                break

    def backprocess(self,target_pixel):
        im_2 = self.im_target.reshape(self.im.shape[0:2]).astype(np.uint8)
        u_labels = np.unique(self.im_target)  # 获取分割后图像的标签类型
        random_sample_num = 50
        image = np.array(self.im)
        image = image.reshape(-1, 3)
        random_sample_pixel_mean = []
        for labels in u_labels:
            pixels_loaction = np.where(self.im_target == labels)[0]
            pixels_loaction = pixels_loaction[np.random.randint(0, len(pixels_loaction), random_sample_num)]
            random_sample_pixel = image[pixels_loaction]
            random_sample_pixel_mean.append(np.mean(random_sample_pixel, axis=0))

        target_pixel = target_pixel.repeat(repeats=len(u_labels), axis=0)
        random_sample_pixel_mean = np.array(random_sample_pixel_mean)
        euclidean_distance = np.linalg.norm(target_pixel - random_sample_pixel_mean, axis=1)
        min_euclidean_distance_loc = euclidean_distance.argmin()
        # if 最小欧氏距离下标 < 30:
        #     print('找到目标像素')
        im_2[im_2 != u_labels[min_euclidean_distance_loc]] = 0
        im_2[im_2 == u_labels[min_euclidean_distance_loc]] = 255
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


    def Seg_image(self,image,target_pixel,device):
        self.im = image
        self.data = torch.from_numpy(
            np.array([self.im.transpose((2, 0, 1)).astype('float32') / 255.]))  # 从numpy.ndarray创建一个张量 transpose表示调换元素位置
        device = device
        self.data = self.data.to(device)

        labels = segmentation.felzenszwalb(self.im, scale=32, sigma=0.5, min_size=64)
        labels = labels.reshape(self.im.shape[0] * self.im.shape[1])
        u_labels = np.unique(labels)
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])
        self.build_model(device)
        self.pre_seg(device)
        self.backprocess(target_pixel)
        return self.im_2, self.im_3

def spot_edge_distance(spot,edge):
    edge_martix = np.full(edge.shape,spot)
    distance = np.linalg.norm(edge_martix - edge, axis=1)
    return distance
def image_quality_control(target_light,target_dark):
    edge, hierarchy = cv2.findContours(target_dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edge_correction = []  #剔除了图像边缘的点
    edge_num = []
    #这里寻找出边缘，除去图像的边缘
    for i in range(len(edge)):
        temp = np.array(edge[i]).reshape(-1, 2)
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
            if temp[j, 1] > target_light.shape[0] - 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        delate_loc = []
        for j in range(temp.shape[0]):
            if temp[j, 0] > target_light.shape[1] - 10:
                delate_loc.append(j)
        temp = np.delete(temp, delate_loc, axis=0)
        edge_correction.append(temp)
        edge_num.append(temp.shape[0])

    if len(edge_num) == 0 or len(edge_num) == 1:
        return False, []
    if np.max(edge_num) < 200:
        return False, []
    if len(edge_correction) == 0 or len(edge_correction) == 1:
        return False, []

    chosen_edage= []
    for edge in edge_correction:
        distance = spot_edge_distance([900, 900], edge)
        if len(distance) == 0:
            return False, []
        diff = np.diff(distance, n=1)
        # 画出边缘(目标暗,边缘点集合,[0,0,0])
        if (abs(distance[0] - distance[-1])) < 50:
            diff = abs(diff)
            if np.max(diff) < 10:
                continue
        chosen_edage.append(edge)

    if len(chosen_edage) == 0:
        return False, []
    return True, chosen_edage
def cut_image(slide, hight=900, width=900, image_level=1, distance_threshold=20, target_pixel=[76.78, 37.92, 109.32]):
    row_num = slide.level_dimensions[image_level][1] // hight
    column_num = slide.level_dimensions[image_level][0] // width
    zoom_scale = slide.level_dimensions[0][1] // slide.level_dimensions[image_level][1]
    ordinate_accumulates_correction = np.zeros(column_num)


    result_img = []
    result_pos = []
    for row in range(row_num):
        abscissa_accumulates_correction = 0
        for column in range(column_num):
            # print("总进度：",now,'/',all,'   ',"进度：",row*列图像数+column,"/",行图像数*列图像数)
            x = column * width * zoom_scale + int(abscissa_accumulates_correction) * zoom_scale
            y = row * hight * zoom_scale + int(ordinate_accumulates_correction[column]) * zoom_scale
            region = slide.read_region((x, y), image_level, (width, hight))
            region = np.array(region)
            b, g, r, a = cv2.split(region)
            crop_image = cv2.merge([r, g, b])
            euclidean_distance = np.linalg.norm(target_pixel - crop_image, axis=2)
            min_euclidean_distance = np.min(euclidean_distance)
            if min_euclidean_distance < distance_threshold:
                similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)
                similarity_spot_num = similarity_spot_loc.shape[0]

                if similarity_spot_num > 5000:
                    abscissa_correction = np.mean(similarity_spot_loc[:, 1]) - width / 2
                    if abscissa_correction < 0:  # 只向前修正，不向后修正，以免重叠
                        _ = 1
                    if abscissa_correction > 0 and abscissa_correction < 100:  # 误差不大，修正后直接结束
                        region = slide.read_region((int(x + abscissa_correction * zoom_scale), y), image_level, (width, hight))
                        x = int(x + abscissa_correction * zoom_scale)
                        y = y
                        region = np.array(region)
                        b, g, r, a = cv2.split(region)
                        crop_image = cv2.merge([r, g, b])
                        abscissa_accumulates_correction += abscissa_correction


                    if abscissa_correction >= 100:  # 误差较大，修正后查看是否有新的像素加入（边缘是否完整）
                        abscissa_correction += 50
                        abscissa_correction_old = abscissa_correction
                        accumulates_correction = abscissa_correction
                        end = 20
                        while True:
                            region = slide.read_region((int(x + accumulates_correction * zoom_scale), y), image_level,
                                                       (width, hight))
                            x = int(x + accumulates_correction * zoom_scale)
                            y = y
                            region = np.array(region)
                            b, g, r, a = cv2.split(region)
                            crop_image = cv2.merge([r, g, b])
                            result_img.append(crop_image)
                            result_pos.append([int(x + accumulates_correction * zoom_scale), y])

                            euclidean_distance = np.linalg.norm(target_pixel - crop_image, axis=2)
                            similarity_spot_loc = np.argwhere(euclidean_distance < distance_threshold)

                            abscissa_correction_new = np.mean(similarity_spot_loc[:, 1]) - width / 2
                            if abscissa_correction_new - abscissa_correction_old < 0 or end == 0:  # 如果变化不大认为边缘已经稳定
                                abscissa_accumulates_correction += accumulates_correction
                                break
                            accumulates_correction += abscissa_correction_new - abscissa_correction_old
                            # print(str(row)+'_'+str(column),np.mean(相似点下标[:,1]),'new',横坐标修正值新,'old',横坐标修正值旧)
                            abscissa_correction_old = abscissa_correction_new
                            end -= 1



                    ordinate_correction = np.mean(similarity_spot_loc[:, 0]) - width / 2
                    if ordinate_correction > 100:  # 误差较大，修正后直接结束
                        region = slide.read_region((int(x), int(y + ordinate_correction * zoom_scale)), image_level,
                                                   (width, hight))
                        x = int(x)
                        y = int(y + ordinate_correction * zoom_scale)
                        region = np.array(region)[:,]
                        b, g, r, a = cv2.split(region)
                        crop_image = cv2.merge([r, g, b])
                        ordinate_accumulates_correction[column] += ordinate_correction
                    result_img.append(crop_image)
                    result_pos.append([x, y])
    return result_img,result_pos


parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True,help='the dir of svg image location(dir path)')
parser.add_argument('-save_path',type=str,required=True,help='the dir of save segmentation result(dir path),it will create two dir 512(save image with 512*512) and 128(save image with 128*128)')
args = parser.parse_args()


# input_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/image'
# save_path  = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/cut_image'

input_path = args.input_path
save_path  = args.save_path

target_pixel = np.array([[76.78, 37.92, 109.32]])
save_path_512 = os.path.join(save_path,'512')
save_path_128 = os.path.join(save_path,'128')

if os.path.exists(save_path_512) == False:
    os.makedirs(save_path_512)
if os.path.exists(save_path_128) == False:
    os.makedirs(save_path_128)

if os.path.exists(save_path) == False:
    os.makedirs(save_path)


#%%

for now,file_name in enumerate(os.listdir(input_path)):
    print(now,'/',len(os.listdir(input_path)),os.path.join(input_path,file_name))
    if now < 300:
        continue
    Image_Slide = openslide.OpenSlide(os.path.join(input_path,file_name))
    result_img, result_pos = cut_image(Image_Slide, 1024, 1024,target_pixel=target_pixel)
    for index in range(len(result_img)):
        tissue_seg = Seg_image()
        target_light, target_dark = tissue_seg.Seg_image(result_img[index],target_pixel,'cuda:0')
        edge_status, edge_list = image_quality_control(target_light, target_dark)
        if edge_status == True:
            ref_x = result_pos[index][0]
            ref_y = result_pos[index][1]
            cut_im = Image_Slide.read_region((int(result_pos[index][0]), int(result_pos[index][1])), 0,(4 * 1024, 4 * 1024))
            cut_im = np.array(cut_im)[:, :, :3]
            for edge_index, edge in enumerate(edge_list):

                edge = 4 * edge
                for loc_index, loc in enumerate(edge[::128]):
                    ref_x = loc[0]
                    ref_y = loc[1]
                    start_x = ref_x - 256
                    end_x = ref_x + 256
                    start_y = ref_y - 256
                    end_y = ref_y + 256
                    if start_x > 0 and end_x < cut_im.shape[0] and start_y > 0 and end_y < cut_im.shape[1]:
                        im = cut_im[start_y:end_y, start_x:end_x, :]
                        target_light, target_dark = tissue_seg.Seg_image(im,target_pixel,'cuda:0')
                        target_light = target_light/255
                        target_light = target_light.astype(int)


                        img_save_dir = os.path.join(save_path_512,file_name,'image')
                        mask_save_dir = os.path.join(save_path_512,file_name,'mask')

                        if os.path.exists(img_save_dir) == False:
                            os.makedirs(img_save_dir)
                        if os.path.exists(mask_save_dir) == False:
                            os.makedirs(mask_save_dir)

                        img_save_path = os.path.join(img_save_dir, '{}{}{}.nii'.format(index, edge_index, loc_index))
                        mask_save_path = os.path.join(mask_save_dir, '{}{}{}.nii'.format(index, edge_index, loc_index))

                        target_light = target_light[:,:,np.newaxis]
                        target_light = np.repeat(target_light,3,2)

                        out = sitk.GetImageFromArray(target_light)
                        sitk.WriteImage(out, mask_save_path)

                        out = sitk.GetImageFromArray(im)
                        sitk.WriteImage(out, img_save_path)

                for loc_index, loc in enumerate(edge[::64]):
                    ref_x = loc[0]
                    ref_y = loc[1]
                    start_x = ref_x - 64
                    end_x = ref_x + 64
                    start_y = ref_y - 64
                    end_y = ref_y + 64
                    if start_x > 0 and end_x < cut_im.shape[0] and start_y > 0 and end_y < cut_im.shape[1]:
                        im = cut_im[start_y:end_y, start_x:end_x, :]
                        target_light, target_dark = tissue_seg.Seg_image(im, target_pixel, 'cuda:0')
                        target_light = target_light / 255
                        target_light = target_light.astype(int)

                        img_save_dir = os.path.join(save_path_128, file_name, 'image')
                        mask_save_dir = os.path.join(save_path_128, file_name, 'mask')

                        if os.path.exists(img_save_dir) == False:
                            os.makedirs(img_save_dir)
                        if os.path.exists(mask_save_dir) == False:
                            os.makedirs(mask_save_dir)

                        img_save_path = os.path.join(img_save_dir,'{}{}{}.nii'.format(index, edge_index, loc_index))
                        mask_save_path = os.path.join(mask_save_dir,'{}{}{}.nii'.format(index, edge_index, loc_index))

                        target_light = target_light[:, :, np.newaxis]
                        target_light = np.repeat(target_light, 3, 2)

                        out = sitk.GetImageFromArray(target_light)
                        sitk.WriteImage(out, mask_save_path)

                        out = sitk.GetImageFromArray(im)
                        sitk.WriteImage(out, img_save_path)


