import random
import os
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_recall_curve,average_precision_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import argparse
from sklearn.metrics import confusion_matrix
#%%
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('-input', help='Data files path',required=True)
parser.add_argument('-output', help='The path where the result willed be saved',required=True)
args = parser.parse_args()
data_path = args.input
save_path = args.output

# data_path = '/mnt/dfc_data2/project/linyusen/database/16_yi_xian_ct_image/window_100_700_adaptive/NODM_T2DM_select_feature_predict_select_normalise_balance0.csv'
# save_path = '/mnt/dfc_data2/project/linyusen/database/16_yi_xian_ct_image/window_100_700_adaptive/NODM_T2DM_RF0'

image_save_path = os.path.join(save_path,'image')
model_save_path = os.path.join(save_path,'model')

if os.path.exists(save_path) == False:
    os.makedirs(save_path)
if os.path.exists(image_save_path) == False:
    os.makedirs(image_save_path)
if os.path.exists(model_save_path) == False:
    os.makedirs(model_save_path)

mat = {}
df = pd.read_csv(data_path ,header = 0,index_col = 0)
df = df.transpose()
cache = []
mat = {}
type = set()

for line in df.iterrows():
    type.add(line[0].split('_')[-1])
type = list(type)
type.sort()
print('lable:',type)
for line in df.iterrows():
    temp = []
    y = type.index(line[0].split('_')[-1])
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
    plt.figure(figsize=(8,8),dpi=100)
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
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=2)
    plt.title(name)
    plt.savefig(os.path.join(image_save_path,'roc_auc.pdf'))
    plt.savefig(os.path.join(image_save_path, 'roc_auc.jpg'))

def draw_PR(result,name,image_save_path):
    plt.figure(figsize=(8,8),dpi=100)
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
    plt.figure(figsize=(3,5))
    plt.boxplot(chose_weight)

    a = [1,2,3,4,5]
    plt.title(name)
    plt.xticks(a,chose_lable,rotation = 90)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.6)
    plt.ylabel('weight')
    plt.savefig(os.path.join(image_save_path,'chose_weight.pdf'))
    plt.savefig(os.path.join(image_save_path, 'chose_weight.jpg'))

def draw_calibration_curve(result,name,image_save_path):
    plt.figure(figsize=(8, 8), dpi=100)
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
        plt.figure(figsize=(5, 5))
        plt.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
        plt.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
        plt.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        plt.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)
        #Figure Configuration， 美化一下细节
        plt.xlim(0,1)
        plt.ylim(-0.2, 0.6)#adjustify the y axis limitation
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
        y_pred_label = np.where(predict_proba > thresh,1,0)
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

#%%
from sklearn.model_selection import StratifiedKFold

def fun_for_RF_train(mat,k_flod=5):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    x = []
    y = []
    for data in mat:
        x.append(data[:-1])
        y.append(data[-1])



    result = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x,y)):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for index in train_idx:
            train_x.append(x[index])
            train_y.append(y[index])
        for index in val_idx:
            test_x.append(x[index])
            test_y.append(y[index])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(train_x, train_y)

        predict_y = model.predict(test_x)
        predict_proba = model.predict_proba(test_x)
        fpr, tpr, roc_auc, acc, precision, recall, AP, fraction_of_positives, mean_predicted_value,thresh_group,\
            net_benefit_model,net_benefit_all = fun_metrics(predict_y, test_y,predict_proba[:,1])
        weights_selected = (model.feature_importances_ ** 2)
        weights_selected /= weights_selected.max()
        weights_selected = list(weights_selected)

        result[fold_idx] = {}
        result[fold_idx]['fpr'] = fpr
        result[fold_idx]['tpr'] = tpr
        result[fold_idx]['roc_auc'] = roc_auc
        result[fold_idx]['acc'] = acc
        result[fold_idx]['weights_selected'] = weights_selected
        result[fold_idx]['precision'] = precision
        result[fold_idx]['recall'] = recall
        result[fold_idx]['AP'] = AP
        result[fold_idx]['model'] = model
        result[fold_idx]['fraction_of_positives'] = fraction_of_positives
        result[fold_idx]['mean_predicted_value'] = mean_predicted_value
        result[fold_idx]['thresh_group'] = thresh_group
        result[fold_idx]['net_benefit_model'] = net_benefit_model
        result[fold_idx]['net_benefit_all'] = net_benefit_all
    return result

result = fun_for_RF_train(mat=mat,k_flod=5)
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