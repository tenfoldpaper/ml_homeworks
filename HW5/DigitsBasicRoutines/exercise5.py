import numpy as np
import math
#import matplotlib.pyplot as plt

#Function for dividing an array into num equal-sized sections
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

m = int(5) #modify the multiplier to get the dissimilarity rate
dataPoints = []
infile = open("mfeat-pix.txt","r")
j = 0
#Parse the first 200 lines
raw = infile.read().splitlines()
for line in raw:
    parsed = line.split('  ')
    parsed[0] = 0
    for i in range(0, 240):
        parsed[i] = int(parsed[i])

    nparray = np.array(parsed[1:])
    nparray = nparray.astype(float)

    if j < 200:
        dataPoints.append(nparray)
        j+=1
#        print("Success!")
infile.close()

#Each element in dataPoints must have length of 240.

N = len(dataPoints)

#print(dataPoints)
#Step 1
def centerDataPoints(x):
    sumX = np.zeros(len(x[0]))
    for i in range(0, len(x)):
        sumX += x[i]
    mean = np.divide(sumX, len(x))
    #print(mean)
    for i in range(0, len(x)):
        x[i] = x[i] - mean
    #print(x[0])
    return mean, x
originalDP = np.matrix(dataPoints).transpose()
mean, cDataPoints = centerDataPoints(dataPoints)
mean = mean.transpose()
dpMatrix = np.matrix(cDataPoints).transpose()
#print(dpMatrix[:,[0]])
###Okay up to here###

#Step 2
C = np.divide((dpMatrix * dpMatrix.transpose()), N)
U, s, V = np.linalg.svd(C)

#print(U.shape)
#print(U[:,[0]].shape)
#print(U[:,[0]].transpose()*dpMatrix[:,[0]])

#get the fk

def sigmaGetter(Uvec, Xmat):
    sigmasq = 0
    #print(len(Uvec))
    for i in range(0, Xmat.shape[1]):
        sigmasq += (Uvec.transpose()*Xmat[:, [i]])**2
    sigmasq /= Xmat.shape[1]
    return sigmasq
sigmas = []

for i in range(0, dpMatrix.shape[0]):
    sigmas.append(sigmaGetter(U[:,[i]],dpMatrix))

#print(sigmas)
print("Done getting sigmas")

#Calculating ratio manually
rnum = 0
rden = 0
for i in range(m, 240):
    rnum += sigmas[i]
for i in range(0, 240):
    rden += sigmas[i]

#Calculating ratio with s from SVD
rnum2 = 0
rden2 = 0
for i in range(m, 240):
    rnum2 += s[i]
for i in range(0, 240):
    rden2 += s[i]

#Those 2 are very, very similar in terms of value
ratio2 = rnum2/rden2
ratio = rnum/rden

#print(rnum, rden)
print("Similarity (1-ratio) for k = {1}: {0}%".format((1-ratio)*100, m))
print("Similarity (1-ratio2) for k = {1}: {0}%".format((1-ratio2)*100, m))

#ratio = rnum/rden
#print("ratio = " ,ratio)
U = U.transpose()
V = V.transpose()
Um = U[:,:m]
Umprime = V[:m,:]

#Compression

cMat = Um.transpose() * originalDP

'''Textbook'''
cMat2 = Umprime * originalDP

#Uncompression
Umfx = Um*cMat
Umfx2 = Umprime.transpose()*cMat2

'''Textbook'''
UmfxTB = Um*cMat2
mean = np.matrix(mean).transpose()

for i in range(0, UmfxTB.shape[1]):
    UmfxTB[:,[i]] = UmfxTB[:,[i]] + mean

for i in range(0, Umfx.shape[1]):
    Umfx[:,[i]] = Umfx[:,[i]] + mean

# decrypt1 = decrypt1.transpose()
# mean = mean.transpose()
# plt.imshow(decrypted, cmap = 'gray')
# plt.show()
