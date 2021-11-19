# coding=utf-8
# 第一部分导入相关包
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split #需要更新sklearn的版本到0.2以上
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from baseclass.SDetection import SDetection

# 第二部分构建模型
class XGBSAD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(XGBSAD, self).__init__(conf, trainingSet, testSet, labels, fold)
    #
    # def buildModel(self):
    #     #定义项目流行度和用户评分密度
    #     self.MUD = {}
    #     #self.RUD = {}
    #     #self.QUD = {}
    #     # computing MUD for training set 训练集
    #     #
    #     #
    #     #
    #     #
    #     # computing MUD for test set 测试集
    #
    #     for user in self.dao.trainingSet_u:
    #
    #
    #
    #
    #  def predict(self):
    #      classifier = XGBClassifier(criterion='entropy')
    #      classifier.fit(self.training, self.trainingLabels)
    #      pred_labels = classifier.predict(self.test)
    #      print('XGBoost:')
    #      return pred_labels



#
# # 导入数据集、划分训练集和测试集
# feature_file = pd.read_excel('../dataset/Weka_Data_1.csv')
#
# x = []# 特征数据
# y = []# 标签
# for index in feature_file.index.values:
#     # print('index', index)
#     # print(feature_file.ix[index].values)
#     x.append(feature_file.ix[index].values[1: -1]) # 每一行都是ID+特征+Label
#     y.append(feature_file.ix[index].values[-1] - 1) #
# x, y = np.array(x), np.array(y)
# print('x,y shape', np.array(x).shape, np.array(y).shape)
# print('样本数', len(feature_file.index.values))
# # 分训练集和测试集
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12343)
# print('训练集和测试集 shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#
# ###############################################################################
# #  交叉验证
# # xgboost
# from xgboost import XGBClassifier
# xgbc_model=XGBClassifier()
#
# # 随机森林
# from sklearn.ensemble import RandomForestClassifier
# rfc_model=RandomForestClassifier()
#
# # ET
# from sklearn.ensemble import ExtraTreesClassifier
# et_model=ExtraTreesClassifier()
#
# # 朴素贝叶斯
# from sklearn.naive_bayes import GaussianNB
# gnb_model=GaussianNB()
#
# #K最近邻
# from sklearn.neighbors import KNeighborsClassifier
# knn_model=KNeighborsClassifier()
#
# #逻辑回归
# from sklearn.linear_model import LogisticRegression
# lr_model=LogisticRegression()
#
# #决策树
# from sklearn.tree import DecisionTreeClassifier
# dt_model=DecisionTreeClassifier()
#
# #支持向量机
# from sklearn.svm import SVC
# svc_model=SVC()
#
# # xgboost
# xgbc_model.fit(x,y)
#
# # 随机森林
# rfc_model.fit(x,y)
#
# # ET
# et_model.fit(x,y)
#
# # 朴素贝叶斯
# gnb_model.fit(x,y)
#
# # K最近邻
# knn_model.fit(x,y)
#
# # 逻辑回归
# lr_model.fit(x,y)
#
# # 决策树
# dt_model.fit(x,y)
#
# # 支持向量机
# svc_model.fit(x,y)
#
# print("\n使用５折交叉验证方法得随机森林模型的准确率（每次迭代的准确率的均值）：")
# print("\tXGBoost模型：",cross_val_score(xgbc_model,x,y,cv=5).mean())
# print("\t随机森林模型：",cross_val_score(rfc_model,x,y,cv=5).mean())
# print("\tET模型：",cross_val_score(et_model,x,y,cv=5).mean())
# print("\t高斯朴素贝叶斯模型：",cross_val_score(gnb_model,x,y,cv=5).mean())
# print("\tK最近邻模型：",cross_val_score(knn_model,x,y,cv=5).mean())
# print("\t逻辑回归：",cross_val_score(lr_model,x,y,cv=5).mean())
# print("\t决策树：",cross_val_score(dt_model,x,y,cv=5).mean())
# print("\t支持向量机：",cross_val_score(svc_model,x,y,cv=5).mean())
#
# ##############################################################################
#
# # 性能评估以XGboost为例
# xgb = xgb.XGBClassifier()
# # 对训练集训练模型
# xgb.fit(X_train,y_train)
# # 对测试集进行预测
# y_pred = xgb.predict(X_test)
# print("\n模型的平均准确率（mean accuracy = (TP+TN)/(P+N) ）")
# print("\tXgboost：",xgb.score(X_test,y_test))
# # print('(y_test,y_pred)', y_test,y_pred)    print("\n性能评价：")
# print("\t预测结果评价报表：\n", metrics.classification_report(y_test,y_pred))
# print("\t混淆矩阵：\n", metrics.confusion_matrix(y_test,y_pred))