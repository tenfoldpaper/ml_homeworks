import numpy as np
import math

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

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

def sigmaGetter(Uvec, Xmat):
    sigmasq = 0
    #print(len(Uvec))
    for i in range(0, Xmat.shape[1]):
        sigmasq += (Uvec.transpose()*Xmat[:, [i]])**2
    sigmasq /= Xmat.shape[1]
    return sigmasq



def pca(dataPoints, m):
    N = len(dataPoints)
    originalDP = np.matrix(dataPoints).transpose()
    mean, cDataPoints = centerDataPoints(dataPoints)
    mean = mean.transpose()
    dpMatrix = np.matrix(cDataPoints).transpose()

    # Step 2
    C = np.divide((dpMatrix * dpMatrix.transpose()), N)
    U, s, V = np.linalg.svd(C)

    sigmas = []

    for i in range(0, dpMatrix.shape[0]):
        sigmas.append(sigmaGetter(U[:,[i]],dpMatrix))

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

    print(Umprime)
    return Umprime
