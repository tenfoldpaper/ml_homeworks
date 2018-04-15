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

MSEtrain = MSEtrain[:71]
MSEtest = MSEtest[:71]
MCtrain = MCtrain[:71]
MCtest = MCtest[:71]

MSEtrainLog = MSEtrainLog[:71]
MSEtestLog = MSEtestLog[:71]
MCtrainLog = MCtrainLog[:71]
MCtestLog = MCtestLog[:71]

x = range(1, 72)

# Grab the index of the minimum value from test mean classification error
testMin = np.argmin(MSEtest)

plt.figure(1)

### Linear plot ###
plt.subplot(211)
plt.plot(x, MSEtrain, 'r--', label='Train mean square error')
plt.plot(x, MCtrain, 'r', label='Train misclassification ratio')
plt.plot(x, MSEtest, 'b--', label='Test mean square error')
plt.plot(x, MCtest, 'b', label='Test misclassification ratio')
plt.plot([testMin, testMin], [0.6, 1], 'g--')
# Marking the minimum point #
plt.annotate('test error min at k={0}'.format(testMin), xy=(testMin, MSEtest[testMin]), xytext=(20, 0.8), arrowprops=dict(facecolor='black', width=1, headwidth=3),)
plt.ylabel('Error')
#plt.xlabel('# of features extracted from PCA')
plt.xlabel('# of features extracted by k-means')
legend = plt.legend(loc = 'upper right', shadow=False)

### Log plot ###
plt.subplot(212)
plt.plot(x, MSEtrainLog, 'r--', label='Train mean square error')
plt.plot(x, MCtrainLog, 'r', label='Train misclassification ratio')
plt.plot(x, MSEtestLog, 'b--', label='Test mean square error')
plt.plot(x, MCtestLog, 'b', label='Test misclassification ratio')
plt.plot([testMin, testMin], [-0.4, 0.05], 'g--')
# Marking the minimum point #
plt.annotate('test error min at k={0}'.format(testMin), xy=(testMin, MSEtestLog[testMin]), xytext=(20, 0.2), arrowprops=dict(facecolor='black', width=1, headwidth=3),)
plt.ylabel('Error in log')
#plt.xlabel('# of features extracted from PCA')
plt.xlabel('# of features extracted by k-means')
legend = plt.legend(loc = 'upper right', shadow=False)


plt.show()
