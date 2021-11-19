# coding=utf-8
import math
import operator


# 读取文件
def readDataSet(dataFile, labelFile):
    dataSet = []
    dataSet0 = []
    data = [0, 0, 0, 0, 0]
    file = open(labelFile, 'r')
    labelList = file.readlines()
    for i in range(5055):
        labelLine = labelList[i].split()
        dataSet0.append([0, 0, 0, 0, 0, labelLine[0], labelLine[1]])
    file.close()
    file = open(dataFile, 'r')
    dataList = file.readlines()
    for i in range(51346):
        dataLine = dataList[i].split()
        for j in range(5055):
            if dataSet0[j][5] == dataLine[0]:
                if dataLine[2] == "1.0":
                    dataSet0[j][0] += 1
                elif dataLine[2] == "2.0":
                    dataSet0[j][1] += 1
                elif dataLine[2] == "3.0":
                    dataSet0[j][2] += 1
                elif dataLine[2] == "4.0":
                    dataSet0[j][3] += 1
                elif dataLine[2] == "5.0":
                    dataSet0[j][4] += 1
            continue
    file.close()
    for i in range(5055):
        sum = dataSet0[i][0] + dataSet0[i][1] + dataSet0[i][2] + dataSet0[i][3] + dataSet0[i][4]
        if sum == 0:
            sum = 1
        for j in range(5):
            data[j] = 100 * dataSet0[i][j] // sum
        dataSet.append([data[0], data[1], data[2], data[3], data[4], dataSet0[i][-1]])
    return dataSet


# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt = shannonEnt - prob * math.log(prob, 2)
    return shannonEnt


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = -1.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy = newEntropy + prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 特征分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 定义分类器
def classify(inputTree, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    if firstStr == '1.0':
        featIndex = 0
    elif firstStr == '2.0':
        featIndex = 1
    elif firstStr == '3.0':
        featIndex = 2
    elif firstStr == '4.0':
        featIndex = 3
    elif firstStr == '5.0':
        featIndex = 4
    if testVec[featIndex] in secondDict.keys():
        closeValue = testVec[featIndex]
    else:
        a = []
        for key in secondDict.keys():
            a.append(abs(testVec[featIndex] - key))
        b = min(a)
        if testVec[featIndex] + b in secondDict.keys():
            closeValue = testVec[featIndex] + b
        else:
            closeValue = testVec[featIndex] - b
    for key in secondDict.keys():
        if closeValue == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 主函数
if __name__ == "__main__":
    dataSet = readDataSet("profiles.txt", "labels.txt")
    labels = ['1.0', '2.0', '3.0', '4.0', '5.0']
    trainDataSet = []
    testDataSet = []
    for i in range(5055):
        if i % 15 != 0:
            trainDataSet.append(dataSet[i])
        else:
            testDataSet.append(dataSet[i])
    fakeUser = 0
    for i in range(len(testDataSet)):
        if testDataSet[i][-1] == "1":
            fakeUser += 1
    decisionTree = createTree(trainDataSet, labels)
    correct = 0.0
    fakeCorrect = 0.0
    fakePredict = 0.0
    for i in range(len(testDataSet)):
        classLabel = classify(decisionTree, testDataSet[i])
        if classLabel == "1":
            fakePredict += 1
        if classLabel == testDataSet[i][-1]:
            correct += 1
            if classLabel == "1":
                fakeCorrect += 1
            print("NO.%d is %s, predict is %s, the predict is correct." % (i, classLabel, testDataSet[i][-1]))
        else:
            print("NO.%d is %s, predict is %s, the predict is wrong." % (i, classLabel, testDataSet[i][-1]))
    correctRate = correct / len(testDataSet)
    precision = fakeCorrect / fakePredict
    recall = fakeCorrect / fakeUser
    f1score = 2 * precision * recall / (precision + recall)
    print("The correct rate is: %f" % correctRate)
    print("The precision ratio is: %f" % precision)  # 精确率
    print("The recall ratio is: %f" % recall)  # 召回率
    print("The f1score is: %f" % f1score)  # f1值
