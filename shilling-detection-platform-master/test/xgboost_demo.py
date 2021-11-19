# -*- coding: utf-8 -*-
# @File    : XGBClassifier_demo.py
### load module
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from xgboost import plot_importance

print("###XGBClassifier_train###")


def show_data(digits):
    ### data analysis and plot data sample
    print(digits.data.shape)
    print(digits.target.shape)
    for i in range(10):
        plt.gray()
        plt.matshow(digits.images[i])
        plt.title(digits.target[i])
        plt.show()


def grid_search_cv(estimator, param_grid):
    # 参数网格搜索法, 选取后可以注释掉
    print("search best parms:")
    grid_search_cv_model = GridSearchCV(estimator,
                                        param_grid, verbose=True)

    # 训练:如果不用grid_search_cv_model可以直接用new的XGBClassifier() model
    clf = grid_search_cv_model.fit(X=digits.data,  # 数据特征
                                   y=digits.target,  # 类别标签
                                   eval_set=[(digits.data, digits.target), ],  # 验证集1
                                   eval_metric="mlogloss",  # 评价损失
                                   early_stopping_rounds=10,  # 连续N次分值不再优化则提前停止
                                   verbose=True  # 和silent参数类似，是否打印训练过程的日志
                                   )
    # 选取最佳参数
    print("Best score: %f using parms: %s" % (clf.best_score_, clf.best_params_))
    return clf.best_params_


def train(estimator, x_train, y_train, x_test, y_test):
    """训练过程"""
    clf = estimator.fit(X=x_train,  # 数据特征
                        y=y_train,  # 类别标签
                        eval_set=[(x_train, y_train),  # 验证集1
                                  (x_test, y_test)],  # 验证集2
                        eval_metric="mlogloss",  # 评价损失
                        early_stopping_rounds=10,  # 连续N次分值不再优化则提前停止
                        verbose=True  # 和silent参数类似，是否打印训练过程的日志
                        )
    # 效果主动输出展示:
    evals_result = model.evals_result()
    print("evals_result: ", evals_result)

    ## plot feature importance
    print("feature_importances: ", clf.feature_importances_)
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model,
                    height=0.5,
                    ax=ax,
                    max_num_features=64)
    plt.show()

    ### make prediction for test data
    y_test_pred = clf.predict(x_test)

    ### model evaluate
    # accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    # recall = recall_score(y_test, y_test_pred)
    # f1 = f1_score(y_test, y_test_pred)
    # print("accuarcy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    # print("recall: %.2f%%" % (recall * 100.0))
    # print("f1: %.2f%%" % (f1 * 100.0))

    print("混淆矩阵:\n", confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    return clf


def save_model(clf, X=None, pk_name="best_boston.pkl"):
    import numpy as np
    # The sklearn API models are picklable
    print("Pickling sklearn API models")
    # must open in binary format to pickle
    pickle.dump(clf, open(pk_name, "wb"))
    clf2 = pickle.load(open(pk_name, "rb"))
    if X is not None:
        print("Verify the model:", np.allclose(clf.predict(X), clf2.predict(X)))
    print("Save model sucess!!")


if __name__ == '__main__':
    grid_search_cv_flag = False  # 是否grid_search_cv
    kfold_flag = False  # 是否进行k折交叉验证

    ### load datasets
    digits = datasets.load_digits()
    show_data(digits)

    ### new一个xgb的分类对象,传入一些超参数
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=100,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=2,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='multi:softmax',  # 指定损失函数
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          silent=1,  # 静默参数,参数值为1时，静默模式开启，不会输出任何信息。默认为0
                          nthread=-1  # 这个参数用来进行多线程控制，应当输入系统的核数。
                          )
    # grid_search_cv
    if grid_search_cv_flag:
        param_grid = {'max_depth': [6, 8], 'n_estimators': [50, 100]}
        best_parms = grid_search_cv(estimator=model, param_grid=param_grid)

    ## train
    # 用最佳参数交叉验证
    print("data split:")
    if kfold_flag:
        kf = KFold(n_splits=2, shuffle=False)  # K折交叉验证
        for train_index, test_index in kf.split(digits.data):
            x_train, y_train = digits.data[train_index], digits.target[train_index]
            x_test, y_test = digits.data[test_index], digits.target[test_index]
            clf = train(estimator=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    else:
        # train_test_split
        x_train, x_test, y_train, y_test = train_test_split(digits.data,
                                                            digits.target,
                                                            test_size=0.3,
                                                            random_state=33)
        clf = train(estimator=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # 保存模型
    save_model(clf=clf, X=x_train)
