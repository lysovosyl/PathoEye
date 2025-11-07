import pandas
import os
import csv
import random
import os
import re
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve,average_precision_score
import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import argparse
from sklearn.metrics import confusion_matrix


input_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/feature_seg'
lable_path = r'/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/label_train_128.tsv'
data_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/pyradiomic_128'
save_path = '/mnt/dfc_data2/project/linyusen/database/07_aging_skin/Sun_Exposed_Lower_leg/finally_test/ml'

# parse = argparse.ArgumentParser()
# parse.add_argument('-input_path',type=str)
# parse.add_argument('-lable_path',type=str)
# parse.add_argument('-data_path',type=str)
# parse.add_argument('-save_path',type=str)
# args = parse.parse_args()
#
# input_path = args.input_path
# lable_path = args.lable_path
# data_path = args.data_path
# save_path = args.save_path


f = open(lable_path)
lines = f.readlines()
file_type = {}
for line in lines:
    line = line[:-1].split('\t')
    file = '-'.join(line[0].split('-')[0:3])
    type = line[1]
    if type not in file_type.keys():
        file_type[type] = []
    file_type[type].append(file)
assert len(file_type.keys()) == 2

#%%
feature = {}
for type in file_type.keys():
    feature[type] = {}
    for file_name in file_type[type]:
        if file_name in os.listdir(data_path):
            for feature_file in os.listdir(os.path.join(data_path,file_name)):
                feature[type]['{}_{}_{}'.format(type,file_name,feature_file.split('.')[0])] = {}
                f = open(os.path.join(data_path,file_name,feature_file))
                reader = csv.reader(f)
                for line in reader:
                    name = line[0]
                    value= float(line[1])
                    feature[type]['{}_{}_{}'.format(type,file_name,feature_file.split('.')[0])][name] = value

import random
min_num = 999999999
for type in feature:
    if len(feature[type]) < min_num:
        min_num = len(feature[type])


#%%
sample_feature = {}
for type in feature:
    for i in random.sample(feature[type].keys(),min_num):
        sample_feature[i] = feature[type][i]

df = pandas.DataFrame(sample_feature)

from scipy.stats import zscore
# df = df.dropna()
df = df.T

mean = df.mean().to_list()
std = df.std().to_list()

df = df.apply(zscore)
df = df.T
df = df.dropna()
df = df.T
#%%
cache = []
mat = {}
type = set()

for line in df.iterrows():
    type.add(line[0].split('_')[0])
type = list(type)
for line in df.iterrows():
    temp = []
    y = type.index(line[0].split('_')[0])
    x = list(line[1])
    temp.extend(x)
    temp.append(y)
    cache.append(temp)
random.shuffle(cache)
random.shuffle(cache)
random.shuffle(cache)
mat = cache


#%%
def save_weight(result,name,save_path):
    weight = []
    for i in result.keys():
        weights_selected = result[i]['weights_selected']
        weight.append(weights_selected)
    df_weight = pd.DataFrame(columns=[df.columns],data=weight)
    df_weight.to_csv(os.path.join(save_path,'{}_weight.csv'.format(name)))

def draw_roc(result,name,image_save_path):
    plt.figure(figsize=(8,8),dpi=500)
    for i in result.keys():
        fpr = result[i]['fpr']
        tpr = result[i]['tpr']
        roc_auc = result[i]['roc_auc']
        plt.plot(fpr, tpr,lw=2, label='ROC curve of '+name+str(i)+'(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    bwith = 3  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=bwith, length=bwith * 2)
    plt.title(name)
    plt.savefig(os.path.join(image_save_path,'roc_auc.pdf'))
    plt.savefig(os.path.join(image_save_path, 'roc_auc.jpg'))

def draw_PR(result,name,image_save_path):
    plt.figure(figsize=(8,8),dpi=500)
    for i in result.keys():
        recall = result[i]['recall']
        precision = result[i]['precision']
        plt.plot(recall, precision,lw=2, label='PR curve of '+name)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=2)
    plt.title(name)
    plt.savefig(os.path.join(image_save_path,'pr.pdf'))
    plt.savefig(os.path.join(image_save_path, 'pr.jpg'))

def draw_weight_hot_point(result,name,image_save_path):
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    weight = []
    for i in result.keys():
        weights_selected = result[i]['weights_selected']
        weight.append(weights_selected)
    plt.figure(figsize=(9,3),dpi=500)
    weight = normalization(weight)
    plt.imshow(weight)
    plt.colorbar()
    plt.xlabel('feature')
    plt.ylabel('k fold num')
    plt.title(name)
    plt.xlabel('feature')
    plt.savefig(os.path.join(image_save_path,'weights_hot_point.pdf'))
    plt.savefig(os.path.join(image_save_path, 'weights_hot_point.jpg'))

def draw_weight_box(result,name,image_save_path):
    every_time_weight = []
    for i in result.keys():
        weights_selected = result[i]['weights_selected']
        every_time_weight.append(weights_selected)
    colume = df.columns
    every_time_weight = np.array(every_time_weight)
    mean_weight = np.mean(every_time_weight,axis=0)
    import matplotlib.pyplot as plt
    chose_weight = []
    chose_lable = []
    for i in np.argsort(mean_weight)[::-1][:5]:
        chose_weight.append(every_time_weight[:,i])
        chose_lable.append(colume[i])
    plt.figure(figsize=(5,8),dpi=500)
    plt.boxplot(chose_weight)

    a = [1,2,3,4,5]
    plt.title(name)
    plt.xticks(a,chose_lable,rotation = 90)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.6)
    plt.ylabel('weight')
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=bwith, length=bwith * 2)
    plt.savefig(os.path.join(image_save_path,'chose_weight.pdf'))
    plt.savefig(os.path.join(image_save_path, 'chose_weight.jpg'))

def draw_calibration_curve(result,name,image_save_path):
    plt.figure(figsize=(8, 8), dpi=500)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i in result.keys():
        fraction_of_positives = result[i]['fraction_of_positives']
        mean_predicted_value = result[i]['mean_predicted_value']
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="Calibration Curve of %s" % (name))

    bwith = 2  # 边框宽度设置为2
    plt.ylim([-0.05, 1.05])
    plt.ylabel("Fraction of positives")
    plt.legend(loc="lower right")
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=2)
    plt.title(name)

    plt.savefig(os.path.join(image_save_path, 'calibration_curve.pdf'))
    plt.savefig(os.path.join(image_save_path, 'calibration_curve.jpg'))
    plt.show()

def draw_DCA(result,name,image_save_path):
    #Plot


    for i in result.keys():
        thresh_group = result[i]['thresh_group']
        net_benefit_model = result[i]['net_benefit_model']
        net_benefit_all = result[i]['net_benefit_all']
        plt.figure(figsize=(5, 5),dpi=500)
        plt.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
        plt.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
        plt.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        plt.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)
        #Figure Configuration， 美化一下细节
        plt.xlim(0,1)
        plt.ylim(-2, 2)#adjustify the y axis limitation
        plt.xlabel(
            xlabel = 'Threshold Probability',
            )
        plt.ylabel(
            ylabel = 'Net Benefit',
            )
        plt.grid('major')
        plt.legend(bbox_to_anchor=(0.8, 1.02), loc=3, borderaxespad=0)
        plt.title(name)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
        plt.savefig(os.path.join(image_save_path, 'Decision_Curve_Analysis_{}.pdf'.format(i)))
        plt.savefig(os.path.join(image_save_path, 'Decision_Curve_Analysis_{}.jpg'.format(i)))
        plt.show()

def fun_metrics(predict,lable,predict_proba):
    acc = accuracy_score(y_true=lable, y_pred=predict)
    print('正确率：',acc)
    fpr, tpr, _ = roc_curve(lable, predict_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(lable, predict_proba)
    fraction_of_positives, mean_predicted_value = calibration_curve(lable, predict_proba)
    AP = average_precision_score(lable, predict_proba, average='macro', pos_label=1, sample_weight=None)

    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = predict > thresh
        tn, fp, fn, tp = confusion_matrix(lable, y_pred_label).ravel()
        n = len(lable)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)

    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(lable, lable).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)


    return fpr,tpr,roc_auc,acc,precision, recall,AP, fraction_of_positives,\
        mean_predicted_value,thresh_group,net_benefit_model,net_benefit_all

def fun_for_RF_train(mat,k_flod=5):
    result = {}
    数据总量 = len(mat)
    切片长度 = int(数据总量 / k_flod)
    数据切片 = []
    for i in range(k_flod):
        数据切片.append(mat[i * 切片长度:(i + 1) * 切片长度])
    for i in range(k_flod):
        result[i] = {}
        test_x = np.array(数据切片[i])
        test_y = np.array(数据切片[i])
        test_x = test_x[:, :-1]
        test_y = test_y[:, -1:]
        test_y = np.reshape(test_y, [-1])

        数据切片_1 = []
        for j in range(k_flod):
            if j != i:
                数据切片_1.extend(数据切片[j])

        train_x = np.array(数据切片_1)
        train_y = np.array(数据切片_1)
        train_x = train_x[:, :-1]
        train_y = train_y[:, -1:]
        train_y = np.reshape(train_y, [-1])

        model = RandomForestClassifier(n_estimators=100)
        model.fit(train_x, train_y)
        predict_y = model.predict(test_x)
        predict_proba = model.predict_proba(test_x)
        fpr, tpr, roc_auc, acc, precision, recall, AP, fraction_of_positives, mean_predicted_value,thresh_group,\
            net_benefit_model,net_benefit_all = fun_metrics(predict_y, test_y,predict_proba[:,1])
        weights_selected = (model.feature_importances_ ** 2)
        weights_selected /= weights_selected.max()
        weights_selected = list(weights_selected)



        result[i]['fpr'] = fpr
        result[i]['tpr'] = tpr
        result[i]['roc_auc'] = roc_auc
        result[i]['acc'] = acc
        result[i]['weights_selected'] = weights_selected
        result[i]['precision'] = precision
        result[i]['recall'] = recall
        result[i]['AP'] = AP
        result[i]['model'] = model
        result[i]['fraction_of_positives'] = fraction_of_positives
        result[i]['mean_predicted_value'] = mean_predicted_value
        result[i]['thresh_group'] = thresh_group
        result[i]['net_benefit_model'] = net_benefit_model
        result[i]['net_benefit_all'] = net_benefit_all
    return result

result = fun_for_RF_train(mat=mat,k_flod=5)


#%%

image_save_path = os.path.join(save_path,'image')
model_save_path = os.path.join(save_path,'model')
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
if os.path.exists(image_save_path) == False:
    os.makedirs(image_save_path)
if os.path.exists(model_save_path) == False:
    os.makedirs(model_save_path)

for i in result.keys():
    f = open(os.path.join(model_save_path,'RF_kfold_{}.pkl'.format(i)), 'wb')
    pickle.dump(result[i]['model'], f)
    f.close()

draw_roc(result,'RF',image_save_path)
draw_weight_box(result,'RF',image_save_path)
draw_PR(result,'RF',image_save_path)
draw_weight_hot_point(result,'RF',image_save_path)
draw_calibration_curve(result,'RF',image_save_path)
draw_DCA(result,'RF',image_save_path)
save_weight(result,'RF',save_path)

df.to_csv(os.path.join(save_path,'feature.csv'))

#%%
import pickle
with open(os.path.join(save_path,'mean.pkl'), 'wb') as file:
    pickle.dump(mean, file)
with open(os.path.join(save_path,'std.pkl'), 'wb') as file:
    pickle.dump(std, file)