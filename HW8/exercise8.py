import numpy as np
from numpy.linalg import *
import pca
import math

def getMCBool(Wopt, featureI, Z):
    receivedIndex = np.argmax((Wopt * featureI)[:-1,0])
    trueIndex = np.argmax(Z[:-1,0])
    return not (receivedIndex == trueIndex)

infile = open("mfeat-pix.txt", "r")

# The labels transformed into binary vectors; 1 at the end for bias
z1 = np.array([1,0,0,0,0,0,0,0,0,0,1]) # first 100
z2 = np.array([0,1,0,0,0,0,0,0,0,0,1]) # second 100
z3 = np.array([0,0,1,0,0,0,0,0,0,0,1]) # third 100 ...
z4 = np.array([0,0,0,1,0,0,0,0,0,0,1])
z5 = np.array([0,0,0,0,1,0,0,0,0,0,1])
z6 = np.array([0,0,0,0,0,1,0,0,0,0,1])
z7 = np.array([0,0,0,0,0,0,1,0,0,0,1])
z8 = np.array([0,0,0,0,0,0,0,1,0,0,1])
z9 = np.array([0,0,0,0,0,0,0,0,1,0,1])
z0 = np.array([0,0,0,0,0,0,0,0,0,1,1])
ztot = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z0]

Z = []
for i in range(0, 10):
    for j in range(0, 100):
        Z.append(ztot[i])

Z = np.matrix(Z).transpose()

allData = []
testData = []
trainData = []
j = 0
k = 0
num = 0

# number of features we want to extract
m = 200

raw = infile.read().splitlines()
print(len(raw))
for line in raw:
    parsed = line.split('  ')
    parsed[0] = 0
    for i in range(0, 240):
        parsed[i] = int(parsed[i])

    nparray = np.array(parsed[1:])
    nparray = nparray.astype(float)

    if j < 100 and num != 10:
        trainData.append(nparray)
        j+=1
        #print(j)

    if j == 100 and k < 100:
        testData.append(nparray)
        k += 1
        #print(k)

    if j == 100 and k == 100:
        j = 0
        k = 0
        print("finished handling number: ", num)
        num += 1

infile.close()

def linearRegression(tr, te, m, Zee):
    trainData = tr
    testData = te
    Z = Zee

    #pca matrix transposed
    mean, pcaMatT = pca.pca(trainData, m)
    #to extract 1st column: pcaMatT[:, 0]

    trainData = np.matrix(trainData).transpose()
    testData = np.matrix(testData).transpose()

    pcaMatT = np.pad(pcaMatT, ((0, 1),(0, 0)), 'constant', constant_values = 1)
    #get compressed training data
    ctData = pcaMatT * trainData
    ctestData = pcaMatT * testData

    Phi = ctData.transpose()


    Wopt = (inv(Phi.transpose() * Phi) * Phi.transpose() * Z.transpose()).transpose()
    print(Wopt.shape)

    SEkTrain = 0
    MRTrain = 0
    SEkTest = 0
    MRTest = 0

    for i in range (0, 1000):
        SEkTrain += pow(norm((Wopt * ctData[:,i] - Z[:,i])[:-1,0]), 2) #Removing the bias

    SEkTrain /= 1000

    for i in range(0, 1000):
        MRTrain += getMCBool(Wopt, ctData[:,i], Z[:, i])

    MRTrain /= 1000.0

    for i in range (0, 1000):
        test = pow(norm((Wopt * ctestData[:,i] - Z[:,i])[:-1,0]), 2) #Removing the bias
        SEkTest += test
        # print(test)

    SEkTest /= 1000
    for i in range(0, 1000):
        MRTest += getMCBool(Wopt, ctestData[:,i], Z[:, i])
    MRTest /= 1000.0

    print SEkTrain, MRTrain
    print SEkTest, MRTest
    return SEkTrain, MRTrain, SEkTest, MRTest

datafile = open("ex8results.txt", "w")

for i in range(1, 241):
    a, b, c, d = linearRegression(trainData, testData, i, Z)
    datafile.write("{0} {1} {2} {3}\n".format(a, b, c, d))

datafile.close()
