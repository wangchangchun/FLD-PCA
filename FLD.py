import numpy as np

import matplotlib.pyplot as plt
import loadData
import random
from numpy.linalg import inv

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def splitXandY(dataset, attrNum, len):

    splitX = np.zeros((len, attrNum))
    splitY = np.zeros((len, 1))

    for i in range(len):
        for j in range(attrNum):
            splitX[i][j] = dataset[i][j]
        splitY[i][0] = dataset[i][attrNum]
    return [splitX, splitY]



class FLD:
    def __init__(self, attrNum, finalDim):
        self.finalDim = finalDim
        self.attrNum = attrNum
        self.X = None
        self.mean = {}
        self.classData = {}


    def initClassData(self):
        temp = {}
        num = 1

        for i, row in enumerate(self.X["data"]):
            if self.X["label"][i][0] not in temp:
                temp[self.X["label"][i][0]] = num
                self.classData[num] = []
                num += 1
            self.classData[temp[self.X["label"][i][0]]].append(row)
        for i in range(1,3):
            self.classData[i] = np.array(self.classData[i])


    def Sb(self):

        self.mean["allData"] = np.array([[np.mean(X_n)] for X_n in self.X['data'].T])
        self.mean["1"] = np.array([[np.mean(X_n)] for X_n in self.classData[1].T])
        self.mean["2"] = np.array([[np.mean(X_n)] for X_n in self.classData[2].T])

        class1Sigma = len(self.classData[1]) * (self.mean["1"] - self.mean['allData'])\
            .dot((self.mean["1"] - self.mean['allData']).T)
        class2Sigma = len(self.classData[2]) * (self.mean["2"] - self.mean['allData']) \
            .dot((self.mean["2"] - self.mean['allData']).T)

        return class1Sigma + class2Sigma


    def Sw(self):
        s_w = np.zeros((self.attrNum, self.attrNum))

        for i in range(1, 3):
            s_i = np.zeros((self.attrNum, self.attrNum))

            for j, entry in enumerate(self.classData[i]):
                #print(entry.dtype, self.mean[str(i)].dtype)
                s_i += (entry[:,None] - self.mean[str(i)]).dot((entry[:,None] - self.mean[str(i)]).T)

            s_w += s_i

        return s_w

    def _calc_k_vec(self):
        Sb = self.Sb()
        Sw = self.Sw()

        # val, vec = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        mat = np.dot(np.linalg.pinv(Sw), Sb)
        val, vec = np.linalg.eig(mat)
        # sort eigen
        eigen_pairs = [(np.abs(val[i]), vec[:, i]) for i in range(len(val))]
        eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)

        # choose top k vectors
        self.k_vec = np.array([eigen_pairs[k][1] for k in range(self.finalDim)])
        # self.k_vec /= np.linalg.norm(self.k_vec, axis=0)

    def reduce(self):
        self._calc_k_vec()
        self.X_f = self.k_vec.dot(self.X["data"].T)
        # print(self.X_f)

    def Jw(self):
        before = np.trace(np.dot(np.linalg.pinv(self.Sb()), self.Sw()))
        up = (self.k_vec .dot(self.Sb()) ).dot(self.k_vec.T)
        down =  (self.k_vec .dot(self.Sw()) ).dot(self.k_vec.T)
        after = up/down

        return before,after

'''
def main():
    splitRatio = 0.67
    dataset = loadData.loadGender()
    # dataset = loadData.loadBreast()
    # dataset = loadData.loadIono()
    # trainingSet, testSet = splitDataset(dataset, splitRatio)
    fld = FLD(1600,2)
    X, y =splitXandY(dataset, 1600,len(dataset))

    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / std
    # print(X_norm)

    fld.X = {
        "label": y
        , "data": X_norm
    }

    fld.initClassData()
    fld.reduce()
    plt.scatter(fld.X_f[0][0:len(fld.classData[1])], fld.X_f[1][0:len(fld.classData[1])], s=5, color='b')
    plt.scatter(fld.X_f[0][len(fld.classData[1]):], fld.X_f[1][len(fld.classData[1]):], s=5, color='r')

    plt.show()


    # plot.ROC(y, result_prob[:, 1])


main()
'''