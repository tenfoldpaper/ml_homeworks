import matplotlib.pyplot as plt

MSEtest = []
MSEtrain = []
MCtest = []
MCtrain = []
inputFile = open("ex8results.txt", "r")
for line in inputFile:
    lineList = line.split(" ")
    MSEtrain.append(float(lineList[0]))
    MCtrain.append(float(lineList[1]))
    MSEtest.append(float(lineList[2]))
    MCtest.append(float(lineList[3]))
x = range(1, 240)
plt.plot(x, MSEtrain, 'r--', x, MCtrain, 'r', x, MSEtest, 'b--', x, MCtest, 'b')
