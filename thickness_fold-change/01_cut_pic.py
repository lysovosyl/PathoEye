import os

import openslide
import cv2
import numpy as np
import shutil
from openslide import deepzoom
import matplotlib.pyplot as plt

#%%
def cut_pic(slide,save_Path,Image_height=900,Image_width=900,Screenshot_level = 1,Distance_Threshold = 20,Target_pixel_color = [76.78, 37.92, 109.32]):
    Number_of_row_images = slide.level_dimensions[Screenshot_level][1] // Image_height
    Number_of_column_mages = slide.level_dimensions[Screenshot_level][0] // Image_width
    Zoom_ratio = slide.level_dimensions[0][1] // slide.level_dimensions[Screenshot_level][1]
    Correction_value_of_vertical_cumulative_scale = np.zeros(Number_of_column_mages)
    for row in range(Number_of_row_images):
        Horizontal_cumulative_standard_correction_value = 0
        for column in range(Number_of_column_mages):
            x = column*Image_width*Zoom_ratio + int(Horizontal_cumulative_standard_correction_value)*Zoom_ratio
            y = row   *Image_height*Zoom_ratio + int(Correction_value_of_vertical_cumulative_scale[column])*Zoom_ratio
            region = slide.read_region((x,y),Screenshot_level,(Image_width,Image_height))
            region = np.array(region)
            b, g,r, a = cv2.split(region)
            Crop_Image = cv2.merge([r, g, b])
            Euclidean_distance = np.linalg.norm(Target_pixel_color - Crop_Image, axis=2)
            Minimum_Euclidean_distance_index = np.min(Euclidean_distance)
            if Minimum_Euclidean_distance_index < Distance_Threshold:
                Similarity_index = np.argwhere(Euclidean_distance < Distance_Threshold)
                Number_of_similarity_points = Similarity_index.shape[0]
                if Number_of_similarity_points>1000:
                    Horizontal_coordinate_correction_value = np.mean(Similarity_index[:, 1]) - Image_width / 2
                    if Horizontal_coordinate_correction_value<0:#只向前修正，不向后修正，以免重叠
                        _=1
                    if Horizontal_coordinate_correction_value>0 and Horizontal_coordinate_correction_value < 100:  # 误差不大，修正后直接结束
                        region = slide.read_region((int(x + Horizontal_coordinate_correction_value*Zoom_ratio ), y), Screenshot_level, (Image_width, Image_height))
                        region = np.array(region)
                        b, g, r, a = cv2.split(region)
                        Crop_Image = cv2.merge([r, g, b])
                        Horizontal_cumulative_standard_correction_value += Horizontal_coordinate_correction_value

                    if Horizontal_coordinate_correction_value >= 100:# 误差较大，修正后查看是否有新的像素加入（边缘是否完整）
                        Horizontal_coordinate_correction_value += 50
                        Old_correction_value_for_horizontal_axis = Horizontal_coordinate_correction_value
                        Accumulated_standard_correction_value   = Horizontal_coordinate_correction_value
                        end_value = 20
                        while True:
                            region = slide.read_region((int(x + Accumulated_standard_correction_value*Zoom_ratio ), y), Screenshot_level, (Image_width, Image_height))
                            region = np.array(region)
                            b, g, r, a = cv2.split(region)
                            Crop_Image = cv2.merge([r, g, b])


                            Euclidean_distance = np.linalg.norm(Target_pixel_color - Crop_Image, axis=2)
                            Similarity_index = np.argwhere(Euclidean_distance < Distance_Threshold)

                            New_correction_value_for_horizontal_axis = np.mean(Similarity_index[:,1]) - Image_width/2
                            if New_correction_value_for_horizontal_axis-Old_correction_value_for_horizontal_axis<0 or end_value==0: #如果变化不大认为边缘已经稳定
                                Horizontal_cumulative_standard_correction_value += Accumulated_standard_correction_value
                                break
                            Accumulated_standard_correction_value += New_correction_value_for_horizontal_axis - Old_correction_value_for_horizontal_axis
                            Old_correction_value_for_horizontal_axis = New_correction_value_for_horizontal_axis
                            end_value -= 1

                    Vertical_coordinate_correction_value = np.mean(Similarity_index[:, 0]) - Image_width / 2
                    if Vertical_coordinate_correction_value > 100:  # 误差较大，修正后直接结束
                        region = slide.read_region((int(x), int(y + Vertical_coordinate_correction_value * Zoom_ratio)), Screenshot_level, (Image_width, Image_height))
                        region = np.array(region)
                        b, g, r, a = cv2.split(region)
                        Crop_Image = cv2.merge([r, g, b])
                        Correction_value_of_vertical_cumulative_scale[column] += Vertical_coordinate_correction_value


                    path = os.path.join(save_Path,Current_file_name)
                    if os.path.exists(path) == False:
                        os.makedirs(path)
                    path = os.path.join(path,str(row)+'_'+str(column)+'.bmp')
                    cv2.imwrite(filename=path,img=Crop_Image)
import sys
# data_path = r'/disk3/zhoujj/project/AI_skin/skin_slides/Sun_Exposed_Lower_leg/'
# save_Path = r'/disk1/linyusen/phology_cut/'


data_path = sys.argv[1]
save_Path = sys.argv[2]



Image_Name_List = os.listdir(data_path)
Image_Name_List.remove('1st.err')
Image_Name_List.remove('2nd.err')
Image_Name_List.remove('2nd.log')
Image_Name_List.remove('dl.sh')
Image_Name_List = Image_Name_List[:]

Abnormal_data = []
Completed_data = []
no_complie = True
while no_complie:
    try:
        for i,Current_file_name in enumerate(Image_Name_List):
            print(i, Current_file_name)
            print("总进度：", i + 1, '/', len(Image_Name_List))
            Current_file_path = os.path.join(data_path,Current_file_name)
            if Current_file_name in Abnormal_data:
                if os.path.exists(Current_file_path):
                    shutil.rmtree(Current_file_path, ignore_errors=True)
                continue
            if Current_file_name in Completed_data:
                continue
            slide = openslide.OpenSlide(Current_file_path)
            cut_pic(slide=slide,
                    save_Path = save_Path,
                    all = len(Image_Name_List),
                    now=i+1)
            if i == len(Image_Name_List)-1:
                print('截图完成')
                no_complie = False
            Completed_data.append(Current_file_name)
    except:
        print("Abnormal_data",Current_file_name)
        Abnormal_data.append(Current_file_name)
        print('重新执行')
for i in Abnormal_data:
    print("Abnormal_data", i)