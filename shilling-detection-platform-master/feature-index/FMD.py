import pandas as pd
import numpy as np
import math
df = pd.read_csv('Scoring_Matrices.csv',index_col=0)
df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)#评分矩阵的转置
print(df)

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
    count = rating_count[1] + rating_count[2] + rating_count[3] + rating_count[4] + rating_count[5]
    Count.append(count)
    # print(rating_count,count)
    for a, b, c, d, e in np.nditer([rating_count[1], rating_count[2], rating_count[3], rating_count[4], rating_count[5]]):
        avg_ratings = (a + 2 * b + 3 * c + 4 * d + 5 * e) / count
        Avg_rating.append(avg_ratings)

# print(Count)
# print(Avg_rating)
###################计算WDA*Ij，存入数组

df1 = pd.read_csv('Scoring_Matrices.csv',index_col=0)
print(df1)
WDA = []  #计算WDA*Ij
for i in range(0,943):
    W = 0
    for j in range(0,1682):
        if df1.iloc[i,j] == 0:
            continue
        w = (math.fabs(df1.iloc[i,j] - Avg_rating[j]))
        W = W + w
    WDA.append(W)

###################计算FMD，并保存
data = np.zeros((943,1))
df2 = pd.DataFrame(data)
df2 = df2.rename(columns={0:'FMD'})
for index,row in df1.iterrows():
    row_count = df1.loc[index].value_counts()#统计每行元素的值的个数
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
    count = rating_count[1] + rating_count[2] + rating_count[3] + rating_count[4] + rating_count[5]
    FMD = WDA[index-1] / count
    print('用户',index,'FMD',FMD)
    df2.iloc[index - 1, 0] = FMD
# print(df2)
# df2.to_csv(r'c:\Users\20795\PycharmProjects\movielens\\FMD.csv')