import os
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xw
import statsmodels.api as sm
import pylab
import sys


lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/lable/hist.txt'
Result_Data_Catalog = r"/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/fold_and_thickness_seg_data"

#%%

tab_dict = {}

Age_mapping = {
'20-29':'20-29',
'30-39':'30-39',
'40-49':'40-49',
'50-59':'50-59',
'60-69':'60-69',
'70-79':'70-79',
}
tab_file = open(lable_path)
lines = tab_file.readlines()
for line in lines:
    line = line.split('\t')
    file_name = line[0].split('-')
    file_name = file_name[0] + '-' + file_name[1] + '-' + file_name[2]
    tab_dict[file_name] = Age_mapping[line[2]]





def Falling_edge_shear_data(data,direction):
    aasd = None
    Crossing_length = int(data.shape[0]*(0.20))
    if direction == 'back':
        data = np.flipud(data)
    difference = np.diff(data)
    Ascending_subscript = np.argwhere(difference>0)
    for iii in Ascending_subscript:
        if data[subscript[0]+Crossing_length:].shape[0]<50:
            return False
        if data[subscript[0]+1] < np.max(data[subscript[0]+Crossing_length:]):
            aasd = subscript[0]
            break
    if direction == 'back':
        aasd = data.shape[0] - subscript[0]
    if aasd ==None:
        return False
    return aasd
#%%
Dataset = {}
for now,dir in enumerate(os.listdir(Result_Data_Catalog)):
    print(now,'/',len(os.listdir(Result_Data_Catalog)))
    Dataset[dir] = {}
    Dataset[dir]['fold'] = []
    Dataset[dir]['thickness'] = []
    for subdir in os.listdir(os.path.join(Result_Data_Catalog,dir)):
        if len(os.listdir(os.path.join(Result_Data_Catalog,dir,subdir))) == 5:
            img = plt.imread(os.path.join(Result_Data_Catalog,dir,subdir,'raw_image.bmp'))
            dark = np.load(os.path.join(Result_Data_Catalog, dir, subdir, 'basecell_dark_000.npy'))
            light = np.load(os.path.join(Result_Data_Catalog, dir, subdir, 'basecell_light_255.npy'))
            thickness_a = open(os.path.join(Result_Data_Catalog, dir, subdir, 'thickness_a.csv'))
            thickness_b = open(os.path.join(Result_Data_Catalog, dir, subdir, 'thickness_b.csv'))
            thickness_a = [float(i) for i in thickness_a.readlines()[0][:-1].split(',')]
            thickness_b = [float(i) for i in thickness_b.readlines()[0][:-1].split(',')]
            thickness_a = np.array(thickness_a)
            thickness_b = np.array(thickness_b)

            Target_area_size = np.sum(dark == 0)
            Target_area_proportion = Target_area_size / (dark.shape[0] * dark.shape[1])
            if Target_area_proportion > 0.15:  # Target_area_proportion过大的，去掉
                continue

            Thickness_difference_a = np.diff(thickness_a, n=1)  # 突变的不要
            Thickness_difference_a = abs(Thickness_difference_a)
            Thickness_difference_b = np.diff(thickness_b, n=1)  # 突变的不要
            Thickness_difference_b = abs(Thickness_difference_b)
            if np.max(Thickness_difference_a) > 20 or np.max(Thickness_difference_b) > 20:
                continue

            if thickness_a.shape[0] < 100 or thickness_b.shape[0] < 100:
                continue

            if np.max(thickness_a) > 200 or np.max(thickness_b) > 200:  # 过厚的不要
                continue

            Value_span = int(thickness_a.shape[0] * 0.10)
            if np.max(thickness_a) == np.max(thickness_a[0:Value_span]):
                subscript = Falling_edge_shear_data(thickness_a, 'forword')
                if subscript == False:
                    continue
                thickness_a = thickness_a[int(subscript):]
            if np.max(thickness_a) == np.max(thickness_a[-Value_span:]):
                subscript = Falling_edge_shear_data(thickness_a, 'back')
                if subscript == False:
                    continue
                thickness_a = thickness_a[:int(subscript)]

            if np.max(thickness_b) == np.max(thickness_b[0:Value_span]):
                subscript = Falling_edge_shear_data(thickness_b, 'forword')
                if subscript == False:
                    continue
                thickness_b = thickness_b[int(subscript):]
            if np.max(thickness_b) == np.max(thickness_b[-Value_span:]):
                subscript = Falling_edge_shear_data(thickness_b, 'back')
                if subscript == False:
                    continue
                thickness_b = thickness_b[:int(subscript)]

            thickness_a_var = np.var(thickness_a)
            thickness_b_var = np.var(thickness_b)

            if np.max([thickness_a_var, thickness_b_var]) > 2000:
                continue

            if thickness_a_var < thickness_b_var:
                Dataset[dir]['thickness'].append(np.average(thickness_a))
            else:
                Dataset[dir]['thickness'].append(np.average(thickness_b))
            Dataset[dir]['fold'].append(np.max([thickness_a_var, thickness_b_var]))


#%%
Remove_duplicate_ages = ['20-29','30-39','40-49','50-59','60-69','70-79']
Variance_data = {}
Thickness_data = {}
Age_thickness = {}
Age_variance = {}
for age in Remove_duplicate_ages:
    Variance_data[age] = []
    Thickness_data[age] = []
    Age_thickness[age] = []
    Age_variance[age] = []

for Patient_ID in Dataset.keys():
    if 'fold' in Dataset[Patient_ID].keys():
        Variance_data[tab_dict[Patient_ID]].extend(Dataset[Patient_ID]['fold'])
    if 'thickness' in Dataset[Patient_ID].keys():
        Thickness_data[tab_dict[Patient_ID]].extend(Dataset[Patient_ID]['thickness'])

for Patient_ID in Dataset.keys():
    if 'fold' in Dataset[Patient_ID].keys():
        if len(Dataset[Patient_ID]['fold']) == 0:
            continue
        Age_variance[tab_dict[Patient_ID]].append(np.average(Dataset[Patient_ID]['fold']))


for Patient_ID in Dataset.keys():
    if 'thickness' in Dataset[Patient_ID].keys():
        if len(Dataset[Patient_ID]['thickness']) == 0:
            continue
        Age_thickness[tab_dict[Patient_ID]].append(np.average(Dataset[Patient_ID]['thickness']))


#%%
for age in Remove_duplicate_ages:
    Variance_data[age] = np.array(Variance_data[age])
    Thickness_data[age] = np.array(Thickness_data[age])
    Age_thickness[age]= np.array(Age_thickness[age])
    Age_variance[age] = np.array(Age_variance[age])

Remove_duplicate_ages = ['20-29','30-39','40-49','50-59','60-69','70-79']
for age in Remove_duplicate_ages:
    print(age,Age_variance[age].shape,Age_thickness[age].shape)


#%%
for age in Remove_duplicate_ages:
    plt.hist(x=Age_thickness[age],  # 指定绘图数据
             bins=100,  # 指定直方图中条块的个数
             color='steelblue',  # 指定直方图的填充色
             edgecolor='black',  # 指定直方图的边框色
             range=[0, 100]
             )
    plt.xlabel('average')
    plt.ylabel('Frequency')
    plt.title(age)
    plt.show()

#%%
Logarithmic_transformation_of_variance_data = Variance_data.copy()
for age in Remove_duplicate_ages:
    Logarithmic_transformation_of_variance_data[age] = np.log(Variance_data[age])
for age in Remove_duplicate_ages:
    plt.hist(x = Logarithmic_transformation_of_variance_data[age], # 指定绘图数据
            bins = 100, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'black', # 指定直方图的边框色
            )
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.title(age)
    plt.show()



#%%
Remove_duplicate_ages = ['20-29','30-39','40-49','50-59','60-69','70-79']
fileName = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/fold_and_thickness_result_data.xlsx'
form = xw.Workbook(fileName)
Subtable = form.add_worksheet("sheet1")
Subtable.activate()
Subtable.write_row('A1',Remove_duplicate_ages)

Subtable.write_column("A2", Variance_data[Remove_duplicate_ages[0]])
Subtable.write_column("B2", Variance_data[Remove_duplicate_ages[1]])
Subtable.write_column("C2", Variance_data[Remove_duplicate_ages[2]])
Subtable.write_column("D2", Variance_data[Remove_duplicate_ages[3]])
Subtable.write_column("E2", Variance_data[Remove_duplicate_ages[4]])
Subtable.write_column("F2", Variance_data[Remove_duplicate_ages[5]])

Subtable.write_column("H2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[0]])
Subtable.write_column("I2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[1]])
Subtable.write_column("J2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[2]])
Subtable.write_column("K2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[3]])
Subtable.write_column("L2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[4]])
Subtable.write_column("M2", Logarithmic_transformation_of_variance_data[Remove_duplicate_ages[5]])

Subtable.write_column("O2", Thickness_data[Remove_duplicate_ages[0]])
Subtable.write_column("P2", Thickness_data[Remove_duplicate_ages[1]])
Subtable.write_column("Q2", Thickness_data[Remove_duplicate_ages[2]])
Subtable.write_column("R2", Thickness_data[Remove_duplicate_ages[3]])
Subtable.write_column("S2", Thickness_data[Remove_duplicate_ages[4]])
Subtable.write_column("T2", Thickness_data[Remove_duplicate_ages[5]])

Subtable.write_column("AA2", Age_thickness[Remove_duplicate_ages[0]])
Subtable.write_column("AB2", Age_thickness[Remove_duplicate_ages[1]])
Subtable.write_column("AC2", Age_thickness[Remove_duplicate_ages[2]])
Subtable.write_column("AD2", Age_thickness[Remove_duplicate_ages[3]])
Subtable.write_column("AE2", Age_thickness[Remove_duplicate_ages[4]])
Subtable.write_column("AF2", Age_thickness[Remove_duplicate_ages[5]])

Subtable.write_column("AH2", Age_variance[Remove_duplicate_ages[0]])
Subtable.write_column("AI2", Age_variance[Remove_duplicate_ages[1]])
Subtable.write_column("AJ2", Age_variance[Remove_duplicate_ages[2]])
Subtable.write_column("AK2", Age_variance[Remove_duplicate_ages[3]])
Subtable.write_column("AL2", Age_variance[Remove_duplicate_ages[4]])
Subtable.write_column("AM2", Age_variance[Remove_duplicate_ages[5]])

form.close()


#%%
for age in Remove_duplicate_ages:
    sm.qqplot(Logarithmic_transformation_of_variance_data[age], line='s')
    pylab.title(age)
    pylab.show()

for age in Remove_duplicate_ages:
    sm.qqplot(Thickness_data[age], line='s')
    pylab.title(age)
    pylab.show()