import numpy as np
import random
import general as g
import loadData
import plot
import FLD
import PCA
import matplotlib.pyplot as plt

class Linear:

    def __init__(self, attrNum, classNum,batchNum):
        self.attrNum = attrNum
        self.batchNum = batchNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])

        self.lr = 0.5
        self.input_node = attrNum
        self.w = np.random.uniform(0,1, (attrNum, 1))

    def train(self, x, y):
        errNum = 0

        output = x.dot(self.w)
        answer = []
        for i in range(len(x)):

            if output[i][0] >= 0:
                answer.append(float(2))
                if y[i][0] == 1:
                    errNum += 1

            else:
                answer.append(float(1))
                if y[i][0] == 2:
                    errNum += 1
        # print(np.shape(answer - y.T))
        de_error = np.array(x.T.dot((answer - y.T).T))
        self.w = self.w - self.lr * de_error

        # print(np.shape(de_error))
        # print("error rate : ")
        # print(errNum / len(x))

        return output

    def predict_test(self,x, y):
        y_prob = []
        # print(np.shape(model.w))
        output = x.dot(self.w)
        max = output.max()
        min = output.min()
        answer = []
        correct = 0
        for i in range(len(y)):
            if output[i][0] >= 0:
                answer.append(float(2))
            else:
                answer.append(float(1))
            if answer[i] == y[i][0]:
                correct += 1
            prob = (output[i][0] - min) / (max - min)
            y_prob.append(prob)

        loss = np.mean((y - output) ** 2)

        # plot.plot_result(x, y,3,5, w=model.w)
        print("accuracy : ")
        print(correct / len(y) * 100)
        confusion_matrix = np.zeros((self.classNum, self.classNum))

        accuracy, confusion_matrix = g.getAccuracy(y, answer, confusion_matrix)

        return y_prob, accuracy




def batch(dataset, batchNum):
    # print(dataset)
    trainSize = int(batchNum)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        new =np.array(copy.pop(index).T).flatten()
        # print(new.shape)
        trainSet.append(new)
    # print(np.array(trainSet).shape)
    return trainSet




def callFLD(dataset, attrNum):
    fld = FLD.FLD(len(dataset[0])-1, 1)
    X, y = g.splitXandY(dataset, attrNum, len(dataset))

    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / std

    fld.X = {
        "label": y
        , "data": X_norm
    }

    fld.initClassData()
    fld.reduce()
    print(fld.Jw())
    m = np.shape(fld.X_f)[1]
    for i in range(m):
        color = ''
        if y[i] == 1: color = 'r'
        if y[i] == 2: color = 'g'
        if y[i] == 3: color = 'b'

        plt.scatter(fld.X_f[0,i], y[i], s=50, c=color)

    plt.show()
    plt.show()

    return np.hstack((fld.X_f.T,y ))

def callPCA(dataset, attrNum, k):
    X, y = g.splitXandY(dataset, attrNum, len(dataset))
    print(k)
    finalData, reconMat = PCA.pca(X, k)

    # PCA.plotBestFit(finalData, reconMat, y)
    return np.hstack((finalData,y )), np.hstack((reconMat, y ))



def main():

    pred_acc = []
    for attrnum in range(1,32):
        model = Linear(attrnum, 2, 30)
        '''
        splitRatio = 0.67
        dataset = loadData.loadIono()

        trainingSet, testSet = g.splitDataset(dataset, 0.67)
        # np.savez('Gender_FLD.npz', train=trainingSet, test=testSet)
        '''
        trainingSet = np.load('Iono.npz')['train']
        testSet = np.load('Iono.npz')['test']

        """
        trainingSet = callFLD(np.array(trainingSet), 32)
        testSet = callFLD(testSet, 32)
    

        """
        trainingSet_2, trainingSet_ori= callPCA(trainingSet, 32, attrnum)
        testSet_2, testSet_ori = callPCA(testSet, 32, attrnum)

        trainingSet = trainingSet_2
        testSet = testSet_2





        for i in range(5000):
            if i % 100 == 0:
                model.lr = model.lr/5
            batchData =batch(trainingSet, model.batchNum)

            x, y = g.splitXandY(batchData, model.attrNum, len(batchData))
            model.train(x, y)

        x, y = g.splitXandY(np.array(testSet), model.attrNum, len(testSet))
        final_output, accuracy = model.predict_test(x, y)
        pred_acc.append(accuracy)
        # plot.ROC(y, final_output)

    plt.plot(range(1,32),pred_acc)
    plt.show()
    return

main()


