import json
import lightgbm as lgb
import pandas as pd
from scipy import io
import h5py
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics
import shap
import FocalLoss
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from FocalLoss import FocalLoss #get the FocalLoss implementation from Halford's blog
from scipy import optimize
from scipy import special
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import hdf5storage
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = h5py.File('demo_trainset.mat)["record_station"] 
dataset = np.transpose(dataset)
dataset = dataset[np.where(np.logical_and(dataset[:, [6]] >0,dataset[:, [6]] <3000))[0],:] 
dataset = shuffle(dataset)
data=dataset[:,[50,5,11,13,14,15,19,32,45,46,47,48,49,25]]
target = dataset[:,[28]]
predset = np.zeros((data.shape[0]))
testset = np.zeros((data.shape[0]))
testx = np.zeros([data.shape[0],data.shape[1]])
predictions = np.zeros((len(target)))
lgb_importance = []
r = []
rmse = []
scores =[]
Accuracy =[]
POD =[]
FAR =[]
CSI =[]
HSS =[]
train_x1 = data
train_y1 = target

for i in range(10):
    split_point = int((244-154)/10*i) + 154
    split_point1 = int((244-154)/10)

    val_x = train_x1[np.where((train_x1[:,0]>=split_point) & (train_x1[:,0]<=split_point + split_point1))[0],:]
    train_x = train_x1[np.where((train_x1[:,0]>split_point + split_point1) | (train_x1[:,0]<split_point))[0],:]
    val_y = train_y1[np.where((train_x1[:,0]>=split_point) & (train_x1[:,0]<=split_point + split_point1))[0],:]
    train_y = train_y1[np.where((train_x1[:,0]>split_point + split_point1) | (train_x1[:,0]<split_point))[0],:]
    
    # The optimization of model setup should refer to the original manuscript, which is not reflected here. 
    # Instantiate Focal loss, an example of model setup for parameter tuning.
    loss = FocalLoss(alpha=0.75, gamma=2)
    
    # An example of parameter setup
    fit_params = {
        'boosting_type': 'gbdt',  # 设置提升类型
        'min_split_gain':0,
        'min_child_weight':1,
        'min_child_samples':200,
        'reg_alpha':0,
        'reg_lambda':10,
        'scale_pos_weight':1,
        'n_estimators':1000,
        'num_leaves': 600,  # 叶子节点数 1200
        'max_depth': -1, # -1
        'learning_rate': 0.005,  # 学习速率 0.05
        'feature_fraction': 0.90,  # 建树的特征选择比例 0.75
        'bagging_fraction': 0.90,  # 建树的样本采样比例 0.75
        'bagging_freq': 1,  # k 意味着每 k 次迭代执行bagging
        'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'is_unbalance' :True,
        'lambda_l1': 0.1,
         'n_jobs':16
    }
    trn_data = lgb.Dataset(train_x, train_y,init_score=np.full_like(train_y, loss.init_score(train_y)))
    val_data = lgb.Dataset(val_x, val_y, reference=trn_data,init_score=np.full_like(val_y, loss.init_score(val_y)))
    clf = lgb.train(fit_params, trn_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=50,
                    valid_names=('fit','val'), fobj=loss.lgb_obj, feval=loss.lgb_eval, verbose_eval=100)  # 训练数据需要参数列表和数据集
    y_pred = special.expit(loss.init_score(train_y) + clf.predict(val_x))
    y_test = val_y.flatten()
    predset[np.where((train_x1[:,0]>=split_point) & (train_x1[:,0]<=split_point + split_point1))[0]] = y_pred
    testset[np.where((train_x1[:,0]>=split_point) & (train_x1[:,0]<=split_point + split_point1))[0]] = val_y.flatten()
    testx[np.where((train_x1[:,0]>=split_point) & (train_x1[:,0]<=split_point + split_point1))[0],:] = val_x
    threshold = 0.42   #Trade-off between POD and FAR is made here
    results=[]
    for pred in y_pred:  
        result = 1 if pred > threshold else 0
        results.append(result)
    
    scores = confusion_matrix(results,val_y)
    FP = scores[1,0]  
    FN = scores[0,1]
    TP = scores[1,1]
    TN = scores[0,0]
    
    Accuracy.append((TP+TN)/(TP+TN+FP+FN))
    POD.append(TP/(TP+FN)) # POD=Recall
    FAR.append(FP/(FP+TP))
    CSI.append(TP/(TP+FP+FN))
    HSS.append(2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)))
    
    print(TP)
    print(POD)
    print(FAR)
    print(CSI)
    print(HSS)
    
clf.save_model('model.txt')