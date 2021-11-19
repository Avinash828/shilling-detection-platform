# coding=utf-8
# xgboost可以加载的数据格式
# 1 libsvm 格式的文本数据；
# 2 Numpy 的二维数组；
# 3 XGBoost 的二进制的缓存文件。加载的数据存储在对象 DMatrix 中

# 思路1：按照其他托攻击检测的算法构建XGBSAD类
# 思路2：按照xgboost糖尿病的案例构建XGBSAD算法

# 导入相关库
import pandas as pd
import numpy as np
import warnings
from random import shuffle
from matplotlib import pyplot as plt
from baseclass.SDetection import SDetection
from sklearn.model_selection import train_test_split  # sklearn版本问题 0.2
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


class XGBSAD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(XGBSAD, self).__init__(conf, trainingSet, testSet, labels, fold)

    # def readConfiguration(self):
    #     pass
    # def printAlgorConfig(self):
    #     pass
    # def initModel(self):
    #     pass

    def buildModel(self):
        self.MUD = {}  # 创建字典MUD key->value user->MUD值
        self.RUD = {}  # 创建字典RUD key->value user->RUD值
        self.QUD = {}  # 创建字典QUD key->value user->QUD值
        # computing MUD,RUD,QUD for training set
        # 对训练集中的项目item进行降序排列，返回新生成的slist，不改变原来列表sort 与 sorted 区别：
        # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
        # list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
        sList = sorted(self.dao.trainingSet_i.iteritems(), key=lambda d: len(d[1]), reverse=True)  # 默认false升序
        maxLength = len(sList[0][1])  # 最大长度 maxLength 这个局部变量程序中没有用到，指的是slist的最大长度
        for user in self.dao.trainingSet_u:  # 外层循环，循环遍历训练集内用户
            self.MUD[user] = 0  # MUD（用户流行度均值）= 系统所有用户已评分项目i的流行度/用户数
            for item in self.dao.trainingSet_u[user]:  # 内层循环
                self.MUD[user] += len(self.dao.trainingSet_i[item])  # / float(maxLength)
            self.MUD[user] / float(len(self.dao.trainingSet_u[user]))  # / 表示浮点数除法，返回浮点结果  //表示整数除法，返回不大于结果的一个最大的整数

            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort(reverse=True)  # 排序 降序
            self.RUD[user] = lengthList[0] - lengthList[-1]  # 计算RUD（用户流行度极差）[0] [-1]-1是最后一个元素

            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 3.0)]  # 计算QUD（用户流行度上四分位数）用户流行度大小升序后在四分之一位置处位置的流行度

        # computing MUD,RUD,QUD for test set
        for user in self.dao.testSet_u:  # 遍历测试集中的用户
            self.MUD[user] = 0  # 同上 计算MUD 用户流行度均值
            for item in self.dao.testSet_u[user]:
                self.MUD[user] += len(self.dao.trainingSet_i[item])  # / float(maxLength)
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]  # 计算RUD（用户流行度极差）
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort()  # 排序 升序
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]  # 计算 QUD（用户流行度上四分位数）

        # preparing examples
        for user in self.dao.trainingSet_u:
            self.training.append([self.MUD[user], self.RUD[user], self.QUD[user]])  # 将 MUD RUD QUD 附加到训练集上
            self.trainingLabels.append(self.labels[user])  # 将标签label附加到训练集标签中

        for user in self.dao.testSet_u:
            self.test.append([self.MUD[user], self.RUD[user], self.QUD[user]])  # 将 MUD RUD QUD 附加到训练集上
            self.testLabels.append(self.labels[user])  # 将标签label附加到测试集标签中

    # def saveModel(self):
    #     pass
    # def loadModel(self):
    #     pass

    def predict(self):
        # classifier = LogisticRegression()
        # classifier.fit(self.training, self.trainingLabels)
        # pred_labels = classifier.predict(self.test)
        # print 'Logistic:'
        # print classification_report(self.testLabels, pred_labels)

        # classifier = SVC()
        # classifier.fit(self.training, self.trainingLabels)
        # pred_labels = classifier.predict(self.test)
        # print('SVM:')
        # print(classification_report(self.testLabels, pred_labels))

        classifier = DecisionTreeClassifier(criterion='entropy')  # 决策树分类器 评价标准 信息熵
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels  # 输出预测标签

        # classifier = XGBClassifier(learning_rate=0.1,
        #                       n_estimators=100,  # 树的个数--1000棵树建立xgboost
        #                       max_depth=6,  # 树的深度
        #                       min_child_weight=2,  # 叶子节点最小权重
        #                       gamma=0.,  # 惩罚项中叶子结点个数前的参数
        #                       subsample=0.8,  # 随机选择80%样本建立决策树
        #                       colsample_btree=0.8,  # 随机选择80%特征建立决策树
        #                       objective='multi:softmax',  # 指定损失函数
        #                       scale_pos_weight=1,  # 解决样本个数不平衡的问题
        #                       random_state=27,  # 随机数
        #                       silent=1,  # 静默参数,参数值为1时，静默模式开启，不会输出任何信息。默认为0
        #                       nthread=-1  # 这个参数用来进行多线程控制，应当输入系统的核数。
        #                       )
        # # classifier.fit(self.training, self.trainingLabels) # 喂给分类器训练集和标签
        # classifier.fit(X=self.training,  # 数据特征
        #                y=self.trainingLabels,  # 类别标签
        #                eval_set=[(self.training, self.trainingLabels), ],  # 验证集1
        #                eval_metric="mlogloss",  # 评价损失
        #                early_stopping_rounds=10,  # 连续N次分值不再优化则提前停止
        #                verbose=True  # 和silent参数类似，是否打印训练过程的日志
        #                )
        # # pred_labels = classifier.predict(self.test) # test_vector
        # pred_labels = classifier.predict(self.test)
        # print('XGBoost:')
        # return pred_labels # 输出预测标签

# 导入数据集

# 切分训练集和测试集

# xgboost模型初始化设置

# # booster:
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'max_depth': 5,
#           'lambda': 10,
#           'subsample': 0.75,
#           'colsample_bytree': 0.75,
#           'min_child_weight': 2,
#           'eta': 0.025,
#           'seed': 0,
#           'nthread': 8,
#           'gamma': 0.15,
#           'learning_rate': 0.01}
#
# # 建模与预测：50棵树
# bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
# ypred = bst.predict(dtest)
#
# # 设置阈值、评价指标
# y_pred = (ypred >= 0.5) * 1
# print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
# print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
# print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
# print('Accuracy: %.4f' % metrics.accuracy_score(test_y, y_pred))
# print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
#
# ypred = bst.predict(dtest)
# print("测试集每个样本的得分\n", ypred)
# ypred_leaf = bst.predict(dtest, pred_leaf=True)
# print("测试集每棵树所属的节点数\n", ypred_leaf)
# ypred_contribs = bst.predict(dtest, pred_contribs=True)
# print("特征的重要性\n", ypred_contribs)

# xgb.plot_importance(bst, height=0.9, title='影响糖尿病的重要特征', ylabel='特征')
# plt.rc('font', family='Arial Unicode MS', size=13)
# plt.show()
