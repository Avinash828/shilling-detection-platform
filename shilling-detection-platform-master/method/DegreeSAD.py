# coding:utf-8

from sklearn.metrics import classification_report
from baseclass.SDetection import SDetection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

class DegreeSAD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(DegreeSAD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def buildModel(self):
        self.MUD = {}
        self.RUD = {}
        self.QUD = {}
        # computing MUD,RUD,QUD for training set
        # 对训练集中的项目进行降序排列，返回新生成的列表slist，
        sList = sorted(self.dao.trainingSet_i.iteritems(), key=lambda d: len(d[1]), reverse=True)
        maxLength = len(sList[0][1])# 最大长度
        for user in self.dao.trainingSet_u:# 外层循环，训练集内用户以此循环遍历
            self.MUD[user] = 0
            for item in self.dao.trainingSet_u[user]:# 内层循环
                self.MUD[user] += len(self.dao.trainingSet_i[item])  # / float(maxLength)
            self.MUD[user] / float(len(self.dao.trainingSet_u[user]))

            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort(reverse=True)# 排序

            # 计算RUD
            self.RUD[user] = lengthList[0] - lengthList[-1]

            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort()

            # 计算QUD
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

        # computing MUD,RUD,QUD for test set
        for user in self.dao.testSet_u:
            self.MUD[user] = 0
            for item in self.dao.testSet_u[user]:
                self.MUD[user] += len(self.dao.trainingSet_i[item])  # / float(maxLength)
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

        # preparing examples
        for user in self.dao.trainingSet_u:
            self.training.append([self.MUD[user], self.RUD[user], self.QUD[user]])
            self.trainingLabels.append(self.labels[user])

        for user in self.dao.testSet_u:
            self.test.append([self.MUD[user], self.RUD[user], self.QUD[user]])
            self.testLabels.append(self.labels[user])

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

        # classifier = XGBClassifier(criterion='entropy')
        # classifier.fit(self.training, self.trainingLabels)
        # pred_labels = classifier.predict(self.test)
        # print('Decision Tree:')
        # return pred_labels

        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels # 输出预测标签
