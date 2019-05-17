import numpy as np
import general as g
import loadData
import plot
import PCA
import FLD
import matplotlib.pyplot as plt

# convert the Iris class to float
class Naive_Bayes:
    def __init__(self, attrNum, classNum):
        self.attrNum = attrNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])


    def separateByClass(self,dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        # print(separated)
        self.prior = g.prior(separated,self.classNum)
        return separated

    def summarizeByClass(self,dataset):
        separated = self.separateByClass(dataset)

        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = g.summarize(instances)
        # print(summaries)
        return summaries

    def predict(self,summaries, inputVector):
        probabilities = g.calculateClassProbabilities(summaries, inputVector)

        prob_parent = 0
        for classValue, probability in probabilities.items():
            # print(probabilities[int(classValue)] * model.prior[0][int(classValue-1)])
            prob_parent += probabilities[int(classValue)] * self.prior[0][int(classValue-1)]
            probabilities[classValue] = probabilities[int(classValue)] * self.prior[0][int(classValue-1)]

        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                probabilities[classValue] /= prob_parent
                bestProb = probability
                bestLabel = classValue
        # print(probabilities)
        return bestLabel, probabilities[2]


    def getPredictions(self,summaries, testSet):
        predictions = []
        prob = []
        for i in range(len(testSet)):
            result, tmp = self.predict(summaries, testSet[i])
            predictions.append(result)
            prob.append(tmp)
        # print(predictions)
        return predictions, prob


def callPCA(dataset, attrNum, k):
    X, y = g.splitXandY(dataset, attrNum, len(dataset))
    print(k)
    finalData, reconMat = PCA.pca(X, k)

    # PCA.plotBestFit(finalData, reconMat, y)
    return np.hstack((finalData,y )), np.hstack((reconMat, y ))

def main():

    orig_attr = 4
    splitRatio = 0.67
    pred_acc = []
    for attrnum in range(2, orig_attr):

        dataset = loadData.loadIris()
        model = Naive_Bayes(attrnum, 3)

        # trainingSet, testSet = g.splitDataset(dataset, splitRatio)
        # np.savez('Iris.npz', train=trainingSet, test=testSet)
        trainingSet = np.load('Iris.npz')['train']
        testSet = np.load('Iris.npz')['test']

        # print(trainingSet.shape)

        trainingSet_2, trainingSet_ori = callPCA(trainingSet, orig_attr, attrnum)
        testSet_2, testSet_ori = callPCA(testSet, orig_attr, attrnum)

        trainingSet = np.array(trainingSet_2)
        testSet = testSet_2

        summaries = model.summarizeByClass(trainingSet)
        # print(summaries)
        predictions,result_prob =  model.getPredictions(summaries, testSet)
        x, y = g.splitXandY(testSet, model.attrNum, len(testSet))
        confusion_dim = len(summaries)
        confusion_matrix = np.zeros((confusion_dim, confusion_dim))
        accuracy, confusion_matrix = g.getAccuracy(testSet, predictions, confusion_matrix)
        print(accuracy)
        pred_acc.append(accuracy)

        plot.ROC(y, result_prob)
    plt.plot(range(2, orig_attr), pred_acc)
    plt.show()
main()

