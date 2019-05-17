import numpy as np
import general as g
import loadData
import plot
import random
import math
import FLD
import PCA
import matplotlib.pyplot as plt

class Bayesian:
    def __init__(self, attrNum, classNum):
        self.attrNum = attrNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])

    def meanVector(self,dataset):
        mean = [g.mean(attribute) for attribute in zip(*dataset)]
        # del summaries[-1]
        # print(mean)
        return mean


    def stdMat(self,dataset):
        std = np.zeros((len(self.meanVector(dataset)), len(self.meanVector(dataset))))
        miu = np.array(self.meanVector(dataset))
        for i in range(len(dataset)):
            diff = (dataset[i] - miu)[:, None]
            std += diff * diff.T
        std /= len(dataset)
        std += np.eye(self.attrNum) * 1e-15
        # print(np.shape(std))
        return std


    def separateByClass(self,dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        # print(separated[2].__len__())
        self.prior = g.prior(separated,self.classNum)
        return separated


    # separate the data by class
    # calculate the each class's mean and standard variance of each attribute (column)
    def summarizeByClass(self,dataset):
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():

            x, y = g.splitXandY(np.array(instances), self.attrNum, len(instances))
            summaries[classValue] = self.meanVector(x), self.stdMat(x)
        return summaries




    def multi_gaussian_pdf(self,x, mean, covar):
        from math import exp, sqrt, log, pi
        from numpy.linalg import inv, det
        n = np.array(mean).shape[0]
        test_data = np.zeros((1, n))
        for i in range(n):
            test_data[0][i] = x[i]
        exp_term = -(1/2) * (test_data-mean).dot(inv(covar)).dot((test_data-mean).T)
        const_term = 1/((2*pi)**(n/2)*(det(covar)**(1/2)))
        # print(const_term * np.asscalar(np.exp(exp_term)) )
        return const_term * np.asscalar(np.exp(exp_term))




    def getPredictions(self,summaries, testSet):
        predictions = []
        result_prob = np.zeros((len(testSet), self.classNum))
        for i in range(len(testSet)):
            parent = 0

            max_prob_class = 0
            best_prob = 0
            for j in range(self.classNum):
                parent += self.multi_gaussian_pdf(testSet[i], summaries[j + 1][0], summaries[j + 1][1]) \
                          * self.prior[0][j]
            for j in range(self.classNum):
                result_prob[i][j] = (self.multi_gaussian_pdf(testSet[i], summaries[j + 1][0], summaries[j + 1][1])
                                    * self.prior[0][j] / parent)
                if result_prob[i][j] > best_prob :
                    best_prob = result_prob[i][j]
                    max_prob_class = j

            predictions.append(max_prob_class+1)
        # print(predictions)
        return predictions , result_prob


def callPCA(dataset, attrNum, k):
    X, y = g.splitXandY(dataset, attrNum, len(dataset))
    print(k)
    finalData, reconMat = PCA.pca(X, k)

    # PCA.plotBestFit(finalData, reconMat, y)
    return np.hstack((finalData,y )), np.hstack((reconMat, y ))

def main():
    splitRatio = 0.67

    pred_acc = []
    for attrnum in range(2,13):

        model = Bayesian(attrnum, 3)
        # dataset = loadData.loadWine()
        # trainingSet, testSet = g.splitDataset(dataset, splitRatio)
        # np.savez('Wine.npz', train=trainingSet, test=testSet)
        trainingSet = np.load('Wine.npz')['train']
        testSet = np.load('Wine.npz')['test']


        trainingSet_2, trainingSet_ori = callPCA(trainingSet, 13, attrnum)
        testSet_2, testSet_ori = callPCA(testSet, 13, attrnum)

        trainingSet =np.array(trainingSet_2)
        testSet = testSet_2
        print(trainingSet.shape)
        summaries = model.summarizeByClass(trainingSet)

        predictions , result_prob = model.getPredictions(summaries, testSet)

        x, y = g.splitXandY(np.array(testSet), model.attrNum, len(testSet))
        # print(x)
        # print(x.shape)
        confusion_matrix = np.zeros((len(summaries), len(summaries)))
        accuracy, confusion_matrix= g.getAccuracy(testSet, predictions, confusion_matrix)
        print(accuracy)
        pred_acc.append(accuracy)
        plot.ROC(y, result_prob[:, 1])
    plt.plot(range(2, 13), pred_acc)
    plt.show()
    return accuracy

main()




