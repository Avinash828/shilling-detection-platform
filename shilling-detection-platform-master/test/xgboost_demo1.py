# coding=utf-8
# 引入基本工具库
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
plt.style.use("ggplot")
from sklearn import datasets

iris = datasets.load_iris()

data = iris.data[:100] # 取前100行
print data.shape

label = iris.target[:100] # 取前100行对应的标签
print label

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

dtrain=xgb.DMatrix(train_x,label=train_y)# 训练x 标签 训练y
dtest=xgb.DMatrix(test_x)# 测试集x 目标是预测标签y
# dtrain = xgb.DMatrix("../dataset/agaricus.txt.test")   # XGBoost的专属数据格式，但是也可以用dataframe或者ndarray
# dtest = xgb.DMatrix("../dataset/agaricus.txt.train")  # # XGBoost的专属数据格式，但是也可以用dataframe或者ndarray

params={'booster':'gbtree',
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
	'max_depth':4,
	'lambda':10,
	'subsample':0.75,
	'colsample_bytree':0.75,
	'min_child_weight':2,
	'eta': 0.025,
	'seed':0,
	'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

# from sklearn import metrics
print 'AUC: %.4f' % metrics.roc_auc_score(test_y,ypred)
print 'ACC: %.4f' % metrics.accuracy_score(test_y,y_pred)
print 'Precision: %.4f' %metrics.precision_score(test_y,y_pred)
print 'Recall: %.4f' % metrics.recall_score(test_y,y_pred)
print 'F1-score: %.4f' %metrics.f1_score(test_y,y_pred)
metrics.confusion_matrix(test_y,y_pred)
