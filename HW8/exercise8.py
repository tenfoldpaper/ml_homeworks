import numpy as np
import pca

infile = open("mfeat-pix.txt", "r")

allData = []
testData = []
trainData = []
j = 0
k = 0
num = 0

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

pcaMat = pca.pca(trainData, 200)

trainData = np.matrix(trainData).transpose()

#get compressed training data 
ctData = pcaMat * trainData