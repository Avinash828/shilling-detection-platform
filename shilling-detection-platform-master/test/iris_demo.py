# coding=utf-8
from sklearn import datasets
import pandas as np
import pandas as pd


iris_datas = datasets.load_iris()
#
# print iris_datas.data  # 数据集中的数据
# print iris_datas.target_name  # iris的种类
#
# iris = pd.DataFrame(iris_datas.data, columns=['SpealLength', 'Spealwidth', 'PetalLength', 'PetalLength'])
#
# # iris.shape
# # iris.head()

from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

style_list = ['o', '^', 's']  # 设置点的不同形状，不同形状默认颜色不同，也可自定义
data = iris_datas.data
labels = iris_datas.target_names
cc = defaultdict(list)

for i, d in enumerate(data):
    cc[labels[int(i / 50)]].append(d)

p_list = []
c_list = []

for each in [0, 2]:
    for i, (c, ds) in enumerate(cc.items()):
        draw_data = np.array(ds)
        p = plt.plot(draw_data[:, each], draw_data[:, each + 1], style_list[i])
        p_list.append(p)
        c_list.append(c)

    plt.legend(map(lambda x: x[0], p_list), c_list)
    plt.title('鸢尾花花瓣的长度和宽度') if each else plt.title('鸢尾花花萼的长度和宽度')
    plt.xlabel('花瓣的长度(cm)') if each else plt.xlabel('花萼的长度(cm)')
    plt.ylabel('花瓣的宽度(cm)') if each else plt.ylabel('花萼的宽度(cm)')
    plt.show()

