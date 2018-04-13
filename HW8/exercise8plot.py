import matplotlib.pyplot as plt
import numpy as np
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

MSEtrainLog = np.log(MSEtrain)
MCtrainLog = np.log(MCtrain)
MSEtestLog = np.log(MSEtest)
MCtestLog = np.log(MCtest)

x = range(1, 241)

testMin = np.argmin(MSEtest)

plt.figure(1)
plt.subplot(211)
plt.plot(x, MSEtrain, 'r--', label='Train mean square error')
plt.plot(x, MCtrain, 'r', label='Train misclassification ratio')
plt.plot(x, MSEtest, 'b--', label='Test mean square error')
plt.plot(x, MCtest, 'b', label='Test misclassification ratio')
plt.plot([testMin, testMin], [0, 3], 'g--')
plt.annotate('test error min', xy=(testMin, MSEtest[testMin]), xytext=(125, 2), arrowprops=dict(facecolor='black', width=1, headwidth=3),)
plt.ylabel('Error')
plt.xlabel('# of features extracted from PCA')

legend = plt.legend(loc = 'upper right', shadow=False)

plt.subplot(212)
plt.plot(x, MSEtrainLog, 'r--', label='Train mean square error')
plt.plot(x, MCtrainLog, 'r', label='Train misclassification ratio')
plt.plot(x, MSEtestLog, 'b--', label='Test mean square error')
plt.plot(x, MCtestLog, 'b', label='Test misclassification ratio')
plt.annotate('test error min', xy=(testMin, MSEtestLog[testMin]), xytext=(125, 1), arrowprops=dict(facecolor='black', width=1, headwidth=3),)

plt.ylabel('Error in log')
plt.xlabel('# of features extracted from PCA')

legend = plt.legend(loc = 'upper right', shadow=False)


plt.show()
