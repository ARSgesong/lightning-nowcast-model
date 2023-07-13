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
x_va=dataset[:,[50,5,11,13,14,15,19,32,45,46,47,48,49,25]]
column_names = ['DOY','HOUR', 'T500', 'SP', 'UW','VW','SH','PM2.5','BC','SS','SO4','OC', 'DUST','Flash']

clf = lgb.Booster(model_file='model.txt')
df = pd.DataFrame(x_va, columns=column_names)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(df)
shap_interaction = explainer.shap_interaction_values(df)
shap.summary_plot(shap_values, df)

# To derive the interaction of variables
shap.dependence_plot('CAPE', shap_values, df.iloc,interaction_index = 12)
shap.summary_plot(shap_interaction, df, max_display=11)