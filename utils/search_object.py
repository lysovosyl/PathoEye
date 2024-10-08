#%%
import matplotlib.pyplot as plt
import openslide
from PIL import Image
import numpy as np
import cv2
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def img_mean_pixel(slide,pos,level,size):
    region = slide.read_region(pos, level, size)
    region = np.array(region)[:,:,:3]
    pixel = np.mean(region, axis=(0, 1))[:3]
    return pixel
def resize_image(img:np.array,size=2048):
    img = Image.fromarray(img)
    width,hight = img.size[:2]
    fold = max(img.size[:2])/size
    new_size = (int(width / fold),int(hight / fold))
    resized_image = img.resize(new_size)
    img = np.array(resized_image)
    return img,fold
# def scan_object(slide,step=500):
#     slide_width,slide_hight = slide.level_dimensions[0]
#     ur_img = img_mean_pixel(slide,(0, 0), 0, (1024,1024))
#     ul_img = img_mean_pixel(slide,(slide_width-1024, 0), 0, (1024,1024))
#     dr_img = img_mean_pixel(slide,(0, slide_hight-1024), 0, (1024,1024))
#     dl_img = img_mean_pixel(slide,(slide_width-1024, slide_hight-1024), 0, (1024,1024))
#     reference_background = np.mean([ur_img,ul_img,dr_img,dl_img], axis=(0))
#     white_background = np.array([250,250,250])
#     reference_hight = step
#     reference_width = step
#     data_count_x = {}
#     for i in range(0,slide_width-reference_width,reference_width):
#         print(i)
#         data_count_x[i] = {}
#         region = slide.read_region((i, 0), 0, (reference_width, reference_hight))
#         region = np.array(region)[:,:,:3]
#         pixel = np.mean(region, axis=(0, 1))
#         for j in range(0,slide_hight-reference_hight,reference_hight):
#             region = slide.read_region((i, j), 0, (reference_width,reference_hight))
#             region = np.array(region)[:,:,:3]
#             pixel = np.mean(region, axis=(0, 1))
#             distance1 = np.linalg.norm(reference_background - pixel)
#             if distance1 > 40:
#                 if j not in data_count_x[i].keys():
#                     data_count_x[i][j] = 0
#                 data_count_x[i][j] += 1
#
#     data_count_y = {}
#     for j in range(0, slide_hight - reference_hight, reference_hight):
#         print(j)
#         data_count_y[j] = {}
#         region = slide.read_region((0, j), 0, (reference_width, reference_hight))
#         region = np.array(region)[:,:,:3]
#         pixel = np.mean(region, axis=(0, 1))
#         for i in range(0, slide_width - reference_width, reference_width):
#             region = slide.read_region((i, j), 0, (reference_width, reference_hight))
#             region = np.array(region)[:,:,:3]
#             pixel = np.mean(region, axis=(0, 1))
#             distance1 = np.linalg.norm(reference_background - pixel)
#             if distance1 > 40:
#                 if i not in data_count_y[j].keys():
#                     data_count_y[j][i] = 0
#                 data_count_y[j][i] += 1
#
#     x = []
#     for i in data_count_x.keys():
#         if len(data_count_x[i]) >= 3:
#             x.append(i)
#     y = []
#     for i in data_count_y.keys():
#         if len(data_count_y[i]) >= 3:
#             y.append(i)
#
#     start_x = max([min(x) - 4*reference_width, 0])
#     start_y = max([min(y) - 4*reference_hight, 0])
#     length_x = min([max(x)-min(x)+8*reference_width,slide_width - start_x])
#     length_y = min([max(y)-min(y)+8*reference_hight,slide_hight - start_y])
#     region = slide.read_region((start_x, start_y), 0, (length_x, length_y))
#     region = np.array(region)
#     img = region[:, :, :3]
#     return img


def scan_object(slide):
    select_lebel = 0
    target_pixel = [174.04746886, 136.90078296, 193.84494564]
    for index,i in enumerate(slide.level_dimensions):
        if i[0] < 5000 or i[1] < 5000:
            select_lebel = index
    fold = slide.level_dimensions[0][0]/slide.level_dimensions[select_lebel][0]
    region = slide.read_region((0, 0), select_lebel, slide.level_dimensions[select_lebel])
    region = np.array(region)
    region = region[:,:,:3]
    distance_target = np.linalg.norm(region - target_pixel,axis=2)
    reference = np.zeros_like(distance_target,dtype=np.uint8)
    reference[distance_target < 100 ] = 1
    kernel = np.ones((50, 50), np.uint8)
    reference = cv2.dilate(reference, kernel, iterations=1)
    kernel = np.ones((100, 100), np.uint8)
    reference = cv2.erode(reference, kernel, iterations=1)
    slide_x,slide_y = slide.level_dimensions[0]
    contours, hierarchy = cv2.findContours(reference, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hole_region_list = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 100:
            shape = np.array(contours[i])
            x_end = min([int(np.max(shape[:,0,0])*fold)+1000,slide_x])
            x_start = max([int(np.min(shape[:, 0, 0])*fold)-1000,0])
            y_end = min([int(np.max(shape[:, 0, 1])*fold)+1000,slide_y])
            y_start = max([int(np.min(shape[:, 0, 1])*fold)-1000,0])
            hole_region = slide.read_region((x_start, y_start), 0, (x_end-x_start, y_end-y_start))
            hole_region = np.array(hole_region)[:,:,:3]
            hole_region_list.append(hole_region)
            break
    return hole_region_list

#%%
if __name__ == '__main__':
    path = '/mnt/dfc_data1/home/linyusen/database/48_pyeye_pic/Zhoujj/20240613类天疱疮/2312315 - 2024-06-13 15.36.02.ndpi'
    slide = openslide.OpenSlide(path)
    img_list = scan_object(slide)
    for img in img_list:
        img,fold = resize_image(img)
        plt.imshow((img))
        plt.show()

