import pandas as pd
import numpy as np
import math
df = pd.read_csv('Scoring_Matrices.csv',index_col=0)
df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)#评分矩阵的转置
# print(df)
############计算每个电影的平均评分
Count = []  #将评分总数和平均分存入数组
Avg_rating = []
for index,row in df.iterrows():
    row_count = df.loc[index].value_counts()#统计每行元素的值的个数
    if 1 not in row_count.index:
        row_count[1] = 0
    if 2 not in row_count.index:
        row_count[2] = 0
    if 3 not in row_count.index:
        row_count[3] = 0
    if 4 not in row_count.index:
        row_count[4] = 0
    if 5 not in row_count.index:
        row_count[5] = 0
    rating_count = row_count[1:]
    count = rating_count[1] + rating_count[2] + rating_count[3] + rating_count[4]+ rating_count[5]
    Count.append(count)
    for a, b, c, d, e in np.nditer([rating_count[1], rating_count[2], rating_count[3], rating_count[4], rating_count[5]]):
        avg_ratings = (a + 2 * b + 3 * c + 4 * d + 5 * e) / count
        Avg_rating.append(avg_ratings)
# print(Count)
# print(Avg_rating)


#############################统计每个用户的总评分次数
df = pd.read_csv('Scoring_Matrices.csv',index_col=0)
# print(df)
S_count = []  #将评分总数和平均分存入数组
for index,row in df.iterrows():
    row_count = df.loc[index].value_counts()#统计每行元素的值的个数
    if 1 not in row_count.index:
        row_count[1] = 0
    if 2 not in row_count.index:
        row_count[2] = 0
    if 3 not in row_count.index:
        row_count[3] = 0
    if 4 not in row_count.index:
        row_count[4] = 0
    if 5 not in row_count.index:
        row_count[5] = 0
    rating_count = row_count[1:]
    count = rating_count[1] + rating_count[2] + rating_count[3] + rating_count[4]+ rating_count[5]
    # print(rating_count,count)
    S_count.append(count)

###################每个用户的最大评分项S_Max,每个用户的最大评分项个数S_Max_count
df = pd.read_csv('Scoring_Matrices.csv',index_col=0)
S_Max = []
S_Max_count = []
for i in range(1,944):
    s = df.loc[i,:]
    s_max = s[s == s.max()].index
    S_Max.append(s_max)
for i in range(0,943):
    S_Max_count.append(len(S_Max[i]))
##################计算用户的其他评分项目（除最高分）个数,Else_count
Else_count = []
for i in range(0,943):
    # print(S_count[i],'-',S_Max_count[i],i+1)
    Else_count.append(S_count[i] - S_Max_count[i])
# print(Else_count)

##################计算属性评分偏移度的平方，存入数组

df1 = pd.read_csv('Scoring_Matrices.csv',index_col=0)
# print(S_Max)
WDA_Square = []
for i in range(0,943):
    W = 0
    for j in range(0,1682):
        if j in S_Max[i]:
            continue
        if df1.iloc[i,j] == 0:
            continue
        w = (df1.iloc[i,j] - Avg_rating[j])**2
        W = W + w
    WDA_Square.append(W)

###################计算FMV，并保存
data = np.zeros((943,1))
df2 = pd.DataFrame(data)
df2 = df2.rename(columns={0:'FMV'})
for i in range(0,943):
    FMV = WDA_Square[i] / Else_count[i]
    print('用户',i + 1,'FMV',FMV)
    df2.iloc[i, 0] = FMV
# print(df2)
# df2.to_csv(r'c:\Users\20795\PycharmProjects\movielens\\FMV.csv')