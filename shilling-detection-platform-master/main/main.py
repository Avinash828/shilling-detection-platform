# coding=utf-8
import sys
import time

sys.path.append("..")# ..代表上一级目录，目的是让解释器搜索到包，预备工作
from SDLib import SDLib # 从SDLib文件中导入SDLib类
from tool.config import Config# tool文件中的config模块导入Config类

if __name__ == '__main__':

    print ('-' * 85)
    print ('Shilling Attack Detection Method Performance Comparison Platform \n')
    print ('(based python 2.7 and pycaret)')
    print ('-' * 85)
    print ('Supervised Methods:')
    print ('1. XGBSAD 2. DegreeSAD   3.CoDetector   4.BayesDetector\n')
    print ('-' * 85)
    print ('Semi-Supervised Methods:')
    print ('5. SemiSAD\n')
    print ('-' * 85)
    print ('Unsupervised Methods:')
    print ('6. PCASelectUsers    7. FAP   8.**timeIndex**\n')
    print ('-' * 85)
    algor = -1
    conf = -1
    order = input('please enter the num of the method to run it:')

    s = time.clock()  # 记录时间

    # if order == 0:
    #     try:
    #         import seaborn as sns
    #     except ImportError:
    #         print '!!!To obtain nice data charts, ' \
    #               'we strongly recommend you to install the third-party package <seaborn>!!!'
    #     conf = Config('../config/visual/visual.conf')
    #     Display(conf).render()
    #     exit(0)

    if order == 1:
        conf = Config('../config/XGBSAD.conf')# .. 代表上一级目录

    elif order == 2:
        conf = Config('../config/DegreeSAD.conf')

    elif order == 3:
        conf = Config('../config/CoDetector.conf')

    elif order == 4:
        conf = Config('../config/BayesDetector.conf')

    elif order == 5:
        conf = Config('../config/SemiSAD.conf')

    elif order == 6:
        conf = Config('../config/PCASelectUsers.conf')

    elif order == 7:
        conf = Config('../config/FAP.conf')

    elif order == 8:
        conf = Config('../config/timeIndex.conf')

    else:
        print ('Error num!')
        exit(-1)

    sd = SDLib(conf)# 实例化sd对象，SDLib是类，相关算法参数传入后，生成sd对象
    sd.execute()# 调用对象sd中的execute方法，执行相关算法
    e = time.clock()  # 再次记录时间
    print ("Run time: %f s" % (e - s))# 输出时间差
