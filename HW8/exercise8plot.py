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
x = range(1, 241)
plt.plot(x, MSEtrain, 'r--', label='Train mean square error')
plt.plot(x, MCtrain, 'r', label='Train misclassification ratio')
plt.plot(x, MSEtest, 'b--', label='Test mean square error')
plt.plot(x, MCtest, 'b', label='Test misclassification ratio')
plt.ylabel('Error')
plt.xlabel('# of features extracted from PCA')

legend = plt.legend(loc = 'upper right', shadow=False)

plt.show()
