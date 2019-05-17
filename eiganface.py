import matplotlib.pyplot as plt
from numpy import prod
from PIL import Image
import numpy as np
import loadData
import random

def splitXandY(dataset, attrNum, len):

    splitX = np.zeros((len, attrNum))
    splitY = np.zeros((len, 1))

    for i in range(len):
        for j in range(attrNum):
            splitX[i][j] = dataset[i][j]
        splitY[i][0] = dataset[i][attrNum]
    return [splitX, splitY]

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def meanVector(dataset):
    mean = [sum(attribute)/float(len(attribute)) for attribute in zip(*dataset)]

    return mean

def vec2img(vec, img_size=(40, 40)):
    return vec.reshape(img_size)


def img2vec(img):
    return img.reshape((prod(img.shape)))



def weight_distance(w1, w2, eigenValues):
    eigenValues = eigenValues[:len(w1)]
    w1 = np.array(w1)
    w2 = np.array(w2)
    diff = (w1 - w2) / eigenValues ** (1 / 2)
    return np.dot(diff, diff)


class EigenFace:
    def __init__(self):
        pass

    def pca(self, X, y, k):
        self.X = X  # (num_data, vec_len)
        self.y = y  # (num_data, )
        m, n  = X.shape[0], X.shape[1]

        self.mean = X.mean(0).T
        avgs = np.tile(np.array([self.mean]).T, (1, m))
        diff_imgs = X.T - avgs
        covX = np.cov(diff_imgs.T)

        featValue, featVec = np.linalg.eig(covX)
        index = featValue.argsort()[::-1]
        featValue = featValue[index]
        featVec = featVec[:, index]
        # First K eigenvector
        featValue = featValue[:k]
        featVec = featVec[:, 0:k]

        # Get EigenFace
        featVec = np.matmul(diff_imgs, featVec)
        W = np.copy(featVec)

        for i in range(k):
            W[:, i] = featVec[:, i] / np.linalg.norm(featVec[:, i])

        W = W.T  # (K, vec_length)
        print(W.shape)
        self.W = W
        self.featValue = featValue


    def predict(self, img, n_eigenvec):
        img = img.copy()
        k = self.W.shape[0]

        # Find input weight
        img_vec = img2vec(img)
        diff_vec = img_vec - self.mean
        diff_face = vec2img(diff_vec)
        input_weights = []
        for i in range(n_eigenvec):
            input_weights.append(np.dot(diff_vec, self.W[i]))

        best_match_img = None
        best_match_error = 1e10
        best_y = 0
        # Compare with training data
        for x, y in zip(self.X, self.y):  # (num_data, vec_len)
            img = vec2img(x)
            img_vec = img2vec(img)
            diff_vec = img_vec - self.mean
            diff_face = vec2img(diff_vec)
            target_weights = []
            for i in range(n_eigenvec):
                target_weights.append(np.dot(diff_vec, self.W[i]))
            error = weight_distance(input_weights, target_weights, self.featValue)
            if error < best_match_error:
                best_match_img = img.copy()
                best_match_error = error
                best_y = y
        return best_match_img, best_y

    def getEiganFace(self,dataset, fileName):
        average = meanVector(dataset)
        eiganface = np.resize(average, (40, 40))

        new_im = Image.fromarray(eiganface)

        if new_im.mode != 'RGB':
            new_im = new_im.convert('RGB')
        new_im.save(fileName + ".bmp")

    def getAllFace(self):
        female = loadData.cutImg("fP1.bmp", 400, 400, False)
        male = loadData.cutImg("mP1.bmp", 400, 400, False)
        faceRegn = loadData.cutImg("facesP1.bmp", 640, 200, False)
        self.getEiganFace(np.array(female), "F")
        self.getEiganFace(np.array(male), "M")
        self.getEiganFace(np.array(faceRegn), "ALL")

dataset =loadData.loadGender()

print(dataset.shape)
trainingSet, testSet = splitDataset(dataset, 0.67)

X,y = splitXandY(np.array(trainingSet),1600,len(trainingSet))

eigenface = EigenFace()
eigenface.pca(X, y, 100)
acc = []
ks = []
X,y = splitXandY(np.array(testSet),1600,len(testSet))
for k in range(1, 100):
    correct = 0
    total = len(X)
    for img_vec, label in zip(X, y):
        img = vec2img(img_vec)

        img_predict, y_pred = eigenface.predict(img, n_eigenvec=k)
        # print(y_pred)
        if y_pred == label:
            correct += 1
    acc.append(correct / total)
    ks.append(k)
plt.plot(ks, acc)
plt.title("Gender")
plt.xlabel("# eigen vector")
plt.ylabel("accuracy")
plt.show()



'''
faceRegn =loadData.loadFace()
print(faceRegn.shape)
trainingSet = faceRegn[0:128,0:-1]
testingSet = faceRegn[128:160:2,:]
print(trainingSet.shape)
print(testingSet)
X,y = splitXandY(testingSet,1600,len(testingSet))

eigenface = EigenFace()
eigenface.pca(trainingSet, faceRegn[0:128,-1], 128)

acc = []
ks = []
for k in range(1, 128):
    correct = 0
    total = len(X)
    for img_vec, label in zip(X, y):
        img = vec2img(img_vec)

        img_predict, y_pred = eigenface.predict(img, n_eigenvec=k)
        if y_pred == label:
            correct += 1
    acc.append(correct / total)
    ks.append(k)
plt.plot(ks, acc)
plt.title("Face Recognition")
plt.xlabel("# eigen vector")
plt.ylabel("accuracy")
plt.show()
eigenface.getAllFace()
'''