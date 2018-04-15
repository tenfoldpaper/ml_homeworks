'''
Machine Learning Exercise 4
Seongjin Bien & Mohit Shreshta

Extension granted by Prof. Jaeger due to my attendance at IK2018.
'''

'''
"It's only a few lines of code." -- Herbert Jaeger, 2018
'''

import random
import numpy as np
from matplotlib import pyplot as plt

#Function for getting the distance between 2 points.
def distance(p1, p2):
    res = np.linalg.norm(p2-p1)
    return res

#Function for comparing 2 numpy array sets
def compareSet(s1, s2, Kv):
    for i in range(0, Kv):
        s1[i].sort(key=np.linalg.norm)
        s2[i].sort(key=np.linalg.norm)
        if (len(s1[i]) != len(s2[i])):
            return False
        for j in range(0, len(s2[i])):
            if(not np.array_equal(s1[i][j], s2[i][j])):
                return False
    return True

#Function for dividing an array into num equal-sized sections
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# ##### Initialization #####
# Minimum = 1000000.0
# dataPoints = []
#
# K = 3 #Change this for different number of K-classes
# infile = open("mfeat-pix.txt","r")
#
# #Parse the first 200 lines
# raw = infile.read().splitlines()
# for line in raw:
#     parsed = line.split('  ')
#     parsed[0] = 0
#     for i in range(0, 240):
#         parsed[i] = int(parsed[i])
#
#     nparray = np.array(parsed[1:])
#     nparray = nparray.astype(float)
#
#     if j < 200:
#         dataPoints.append(nparray)
#         j+=1
# #        print("Success!")
# infile.close()

# #Shuffle the resulting vector
# random.shuffle(dataPoints)
# S = chunkIt(dataPoints, K)

##### Initialization END #####
def kMeans(dataPoints, K):
    ##### Repetition #####
    Minimum = 1000000.0 #huge value for initial min value
    random.shuffle(dataPoints)
    S = chunkIt(dataPoints, K)
    iterations = 0
    while(1):
        sMeans = []

        for k in range(0,K):

        #initialize the mean vector
            vecsum = np.zeros((240,), dtype=np.float64)
            j = 0
            i = 0

        #Get the sum of all vectors in set Sn
            for j in range(0,len(S[k])):
                vecsum += S[k][j]

            #Divide vecsum by # of vectors in Sn
            vecsum = np.divide(vecsum, len(S[k]))

            sMeans.append(vecsum)

        #initialize S prime
        # print("Done calculating means")
        Sprime = []
        for i in range(0,K):
            Sprime.append([])

        for i in range(0, len(dataPoints)):
            currentMin = Minimum
            currentMinIndex = 0
            for j in range(0, K):
                if(distance(dataPoints[i], sMeans[j]) < currentMin):
                    currentMin = distance(dataPoints[i], sMeans[j])
                    currentMinIndex = j
            Sprime[currentMinIndex].append(dataPoints[i])


        #Compare S and Sprime
        if(compareSet(S, Sprime, K)):
                # print("Normal of mean vector {0}".format(i))
                # print(np.linalg.norm(sMeans[i]))
            # print("All equal now.")
            break

        #See if there are any empty sets
        count = 0
        while(count < K ):
            if(len(Sprime[count]) == 0):
                del Sprime[count]
                # print("Deleting redundancy")
                K -= 1
            count += 1

        #Set S = S'
        for i in range(0, K):
            S[i] = Sprime[i]

        # print("Let's go for another round!")
        iterations += 1

    return np.matrix(sMeans) #return the codebook vectors as a matrix
    ##### Repetition END #####
#
# ##### Visualize #####
# print("done")
# print("Total iterations: {0}".format(iterations))
