# coding=utf-8
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

profiles_dataset = loadtxt("../dataset/ml-100K/profiles.txt", delimiter=None)
labels_dataset = loadtxt("../dataset/ml-100K/labels.txt", delimiter=None)
# 分为数据集和标签集
X = profiles_dataset
Y = labels_dataset

seed = 6  # 指定一个随机种子，保证每次分割都一致
test_size = 0.2  # 训练集和验证集的分割比例
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy * 100.0))

