# !/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'QiuZiXian'  http://blog.csdn.net/qqzhuimengren/   1467288927@qq.com
# @time          :2020/9/14  0:07
# @abstract    :

import pandas as pd


# 粗略查看数据信息
u_data = pd.read_csv('../dataset/ml-100K/u.data')
u_genre = pd.read_csv('../dataset/ml-100K/u.genre')
u_info = pd.read_csv('../dataset/ml-100K/u.info')
#u_item = pd.read_csv('D:/d/python/ml-100k/u.item')
u_occupation = pd.read_csv('/ml-100K/u.occupation')
u_user = pd.read_csv('/ml-100K/u.user')

print(u_data.head())
print(u_user.head())


# 去掉occupation为none的记录
nones = u_user[u_user['occupation'] == 'none']
u_user = u_user.drop(nones.index)

# gender中的m、f映射成0、 1
u_user['gender'] = u_user['gender'].map({'M':1, 'F':0})
print(u_user.head())


# 对age进行分段，映射成7组
def age_map(age):
    if age >= 1 and age <= 7: return 1
    if age >= 8 and age <=16: return 2
    if age >=17 and age <= 29: return 3
    if age >= 30 and age <= 39: return 4
    if age >= 40 and age <= 49: return 5
    if age >= 50 and age <= 59: return 6
    if age >= 60: return 7

u_user['age'] = u_user['age'].apply(lambda age : age_map(age))
print(u_user.head())

#  occupation字段数值化
def occupations_map(occupation):
    occupations_dict = {'technician': 1,
     'other': 0,
     'writer': 2,
     'executive': 3,
     'administrator': 4,
     'student': 5,
     'lawyer': 6,
     'educator': 7,
     'scientist': 8,
     'entertainment': 9,
     'programmer': 10,
     'librarian': 11,
     'homemaker': 12,
     'artist': 13,
     'engineer': 14,
     'marketing': 15,
     'none': 16,
     'healthcare': 17,
     'retired': 18,
     'salesman': 19,
     'doctor': 20}
    return occupations_dict[occupation]
u_user['occupation'] = u_user['occupation'].apply(lambda occupation : occupations_map(occupation))
print(u_user.head())
#  zip_code提取前3位
u_user['zip_code'] = u_user['zip_code'].apply(lambda zip_code : str(zip_code)[:3])
# 处理好的数据保存，留待后续直接使用
u_user.to_csv('D:/d/python/u_result.csv')