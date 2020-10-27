import os
import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('libsvm-3.21\python')
from svmutil import *
from numpy import *


filename = 'krkopt.data'
fr = open(filename,encoding='utf-8-sig')
arrayOLines = fr.readlines()
numberOfLines = len(arrayOLines)
numberOfFeatureDimension = 6
data = zeros((numberOfLines, numberOfFeatureDimension))  # prepare matrix to return
label = zeros(numberOfLines)    #zeros把其中所有数据用零填入
index = 0
for line in arrayOLines:
    line = line.strip()
    listFromLine = line.split(',')
    if listFromLine[0] =='??':
        break
    data[index, 0] = ord(listFromLine[0])-96
    data[index, 1] = ord(listFromLine[1]) - 48
    data[index, 2] = ord(listFromLine[2])-96
    data[index, 3] = ord(listFromLine[3]) - 48
    data[index, 4] = ord(listFromLine[4])-96
    data[index, 5] = ord(listFromLine[5]) - 48
    if listFromLine[-1] == 'draw':
        label[index] = 1
    else:
        label[index] = -1
    index = index +1


permutatedData = zeros((numberOfLines, numberOfFeatureDimension))
permutatedLabel = zeros(numberOfLines)# p
p = random.permutation(numberOfLines)   #随机排列一个数组或序列
#随机的将数据填入permutatedData并填入对应的标签
for i in range(numberOfLines):
    permutatedData[i,:] = data[p[i],:]
    permutatedLabel[i] = label[p[i]]
#选择其中随机的5000个数据作为样本
numberOfTrainingData = 5000
averageData = zeros((1,numberOfFeatureDimension ))

#求出每个维度的均值和方差
#将需要置换的数据累加并求出平均值
for i in range(numberOfTrainingData):
    averageData +=permutatedData[i,:]

averageData = averageData/numberOfTrainingData

standardDeviation = zeros((1,numberOfFeatureDimension ))

#经验：样本归一化
#计算方差
for i in range(numberOfTrainingData):
    standardDeviation+=(permutatedData[i, :]-averageData[0,:])**2

standardDeviation = (standardDeviation/(numberOfTrainingData-1))**0.5

#在训练和测试样本上同时进行归一化
for i in range(numberOfLines):
    permutatedData[i,:] = (permutatedData[i,:] -averageData[0,:])/standardDeviation[0,:]


y = []
x = []

for i in range(numberOfLines):
    y += [float(permutatedLabel[i])]
    xi = {}
    for ind in range(numberOfFeatureDimension ):    #range()从零开始，且取不到边界值
       xi[int(ind+1)] = permutatedData[i,ind]   #ind+1？
    x += [xi]   #这里的x是一个列表而不是累加

#C是平衡最大化的间隔
#gamma是高斯核函数的一个参数
CScale = [-5, -3, -1, 1, 3, 5,7,9,11,13,15]
gammaScale = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
maxRecognitionRate = 0
for i in range(len(CScale)):
    testC = 2 ** CScale[i]
    for j in range(len(gammaScale)):
        cmd = '-t 2 -c '    #-t是选择核函数，默认值为2,2为高斯核函数 -c是设置SVR中从惩罚系数C，默认值为1
        cmd += str(testC)
        cmd += ' -g '       #设置核函数中γ的值，默认为1/k，k为特征（或者说是属性）数；
        testGamma = 2**gammaScale[j]
        cmd += str(testGamma)
        cmd += ' -v 5'      #-v~n 表示n折验证模式，这里系数为5表示五折验证模式
        recognitionRate = svm_train(y[:numberOfTrainingData], x[:numberOfTrainingData], cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            #print(maxRecognitionRate)
            maxCIndex = i
            maxGammaIndex = j

# 生成新的关于C和gamma的矩阵
'''
缩小C和gamma的取值范围，减小步长，使他的范围减少，参数数据量不减少
因为上面CScale的范围为线性递增，所以以maxCIndex为分界点，取maxCIndex左边的最大值和右边的最小值来提升精度
'''
n = 10;
minCScale = 0.5*(CScale[max(0,maxCIndex-1)]+CScale[maxCIndex])
maxCScale = 0.5*(CScale[min(len(CScale)-1,maxCIndex+1)]+CScale[maxCIndex])
newCScale = arange(minCScale,maxCScale+0.001,(maxCScale-minCScale)/n)

minGammaScale = 0.5*(gammaScale[max(0,maxGammaIndex-1)]+gammaScale[maxGammaIndex])
maxGammaScale = 0.5*(gammaScale[min(len(gammaScale)-1,maxGammaIndex+1)]+gammaScale[maxGammaIndex])
newGammaScale = arange(minGammaScale,maxGammaScale+0.001,(maxGammaScale-minGammaScale)/n)

maxRecognitionRate = 0
for testCScale in newCScale:
    testC = 2 ** testCScale
    for testGammaScale in newGammaScale:
        testGamma = 2**testGammaScale
        cmd = '-t 2 -c '
        cmd += str(testC)
        cmd += ' -g '
        cmd += str(testGamma)
        cmd += ' -v 5'
        recognitionRate = svm_train(y[:numberOfTrainingData], x[:numberOfTrainingData], cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            # print(maxRecognitionRate)
            maxC = testC
            maxGamma = testGamma

#maxC = 1024
#maxGamma = 0.0078125
cmd = '-t 2 -c '
cmd += str(maxC)
cmd += ' -g '
cmd += str(maxGamma)
model = svm_train(y[:numberOfTrainingData], x[:numberOfTrainingData],cmd)

# save variables
pickle.dump([x,y,numberOfTrainingData,maxC,maxGamma],open("mydata","wb")) #序列化对象，将对象obj保存到文件中

x,y,numberOfTrainingData,maxC,maxGamma = pickle.load(open("mydata","rb")) #反序列化对象，将文件中的数据解析为一个python对象
cmd = '-t 2 -c '
cmd += str(maxC)
cmd += ' -g '
cmd += str(maxGamma)
model = svm_train(y[:numberOfTrainingData], x[:numberOfTrainingData],cmd)
labels, accuracy, decisionValues = svm_predict(y[numberOfTrainingData:], x[numberOfTrainingData:], model)
#draw ROC
sortedDecisionValues = sorted(decisionValues)
sortedIndex = sorted(range(len(decisionValues)), key=decisionValues.__getitem__)
sortedLabels = zeros(sortedIndex.__len__())
for i in range(sortedIndex.__len__()):
    sortedLabels[i] = labels[sortedIndex[i]]
truePositive = zeros(sortedIndex.__len__()+1)
falsePositive = zeros(sortedIndex.__len__()+1)
for i in range(sortedIndex.__len__()):
    if sortedLabels[i]>0.5:
        truePositive[0] +=1
    else:
        falsePositive[0] +=1
for i in range(sortedIndex.__len__()):
    if sortedLabels[i]>0.5:
        truePositive[i + 1] = truePositive[i] - 1
        falsePositive[i + 1] = falsePositive[i]
    else:
        falsePositive[i + 1] = falsePositive[i] - 1
        truePositive[i + 1] = truePositive[i]
truePositive = truePositive/truePositive[0]
falsePositive = falsePositive/falsePositive[0]
newTruePositive =  np.arange(0,1,0.01)  #!!!np.arange(1,-0.01,0)
newFalsePositive = zeros(len(newTruePositive))
newFalsePositive = zeros(sortedIndex.__len__()+1)
flag = 0
for i in range(len(truePositive)):
    totalFalsePositive = 0
print(truePositive)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(falsePositive,truePositive)
plt.show()

'''
iter:迭代次数
nu：选择的核函数的类型的参数
obj：SVM文件转换为的二次规划求解得到的最小值
rho：裁决函数的偏执项b
nSV:标准支持向量个数介于[0,C]
nBSV:边界上的支持向量个数(a[i]=C)
Total nSV:支持向量总个数（对于两类来说，因为只有一个分类模型Total nSV = nSV，但是对于多类，这个是各个分类模型的nSV之和
Cross Validation Accuracy:交叉验证精度
'''