import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import loadData



def meanVector(dataset):
    mean = [sum(attribute)/float(len(attribute)) for attribute in zip(*dataset)]

    return mean

def stdMat(dataset,attrNum):
    std = np.zeros((len(meanVector(dataset)), len(meanVector(dataset))))
    miu = np.array(meanVector(dataset))
    for i in range(len(dataset)):
        diff = (dataset[i] - miu)[:, None]
        std += diff * diff.T
    std /= len(dataset)
    std += np.eye(attrNum) * 1e-15
    # print(np.shape(std))
    return std

def splitXandY(dataset, attrNum, len):

    splitX = np.zeros((len, attrNum))
    splitY = np.zeros((len, 1))

    for i in range(len):
        for j in range(attrNum):
            splitX[i][j] = dataset[i][j]
        splitY[i][0] = dataset[i][attrNum]
    return [splitX, splitY]


def pca(XMat, k):
    # print(XMat)
    average = meanVector(XMat)
    # print(average)
    m, n = np.shape(XMat)

    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs

    covX = stdMat(data_adjust,len(XMat[0])) #計算協方差矩陣

    featValue, featVec= np.linalg.eig(covX) #求解協方差矩陣的特徵值和特徵向量
    index = np.argsort(-featValue) #按照featValue進行從大到小排序
    finalData = []

    if k > n:

        print ("k is bigger  than feature num!!")
        return

    else:

        selectVec = np.array(featVec.T[index[:k]]) #所以這裡需要進行轉置

        finalData = data_adjust.dot(selectVec.T)

        reconData = (finalData.dot(selectVec)) + average



        return finalData, reconData


def plotBestFit(data1, data2, y):

    dataArr1 = np.array(data1)

    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]



    for i in range(m):
        color = ''
        if y[i]==1: color = 'r'
        if y[i] == 2: color = 'g'
        if y[i] == 3: color = 'b'

        plt.scatter(dataArr1[i, 0], dataArr1[i, 1], s=50, c=color)


    plt.show()

'''

dataset = loadData.loadIris()
# print(dataset)

X,y = splitXandY(dataset,4,len(dataset))
# print(X)
finalData, reconMat = pca(X, 2)
print(finalData.shape)
print(reconMat.shape)
plotBestFit(finalData, reconMat,y)


'''



