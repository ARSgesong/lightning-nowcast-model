#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:58:29 2023

@author: sg
"""
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

# due to the data storage limit, a demo for prediction result with 2,000,000 predictions are demonstrated here
dataset = h5py.File('demo_prediction.mat')["applyset"] 
dataset = np.transpose(dataset)

testset = dataset[:,14]
predset = dataset[:,15]

precision, recall, thresholds = precision_recall_curve(testset, predset, pos_label=1)
plt.plot(recall, precision, lw=3)
plt.xlabel("Recall", fontsize=17)
plt.ylabel("Precision", fontsize=17)
plt.legend(fontsize=14, loc="upper right")
auc(recall,precision)

false_positive_rate, true_positive_rate, thresholds = roc_curve(testset, predset)
plt.plot(true_positive_rate, false_positive_rate, lw=3)
plt.xlabel("true positive rate", fontsize=17)
plt.ylabel("false positive rate", fontsize=17)
plt.legend(fontsize=14, loc="upper right")
auc(false_positive_rate, true_positive_rate)

