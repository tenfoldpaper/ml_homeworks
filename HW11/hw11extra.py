import numpy as np
import matplotlib.pyplot as plt
import random
# sigmoid
def tanh(a):
    return np.tanh(a)
    
def tanh_derived(a):
    return 1.0 - np.tanh(a)**2

def output_process(x):
    if(x<0):
        return -1
    elif(x==0):
        return 0
    else:
        return 1

def process_error(p):
    if(p>1):
        return 1
    else:
        return p

#Returns a function vectorizedSigmoid which takes np arrays in
#and returns nparrays too
vectorizedTanh = np.vectorize(tanh)
vectorizedTanhDeriv = np.vectorize(tanh_derived)

def backprop(x, y, W1, W2, learn_rate = .005):
    #Forward pass
    l1 = np.matrix(x, dtype = 'f')
    l1 = np.c_[np.ones(1), l1] #adding bias

    #Equations 47 and 48
    l2 = np.c_[np.ones(1), vectorizedTanh(l1.dot(W1))]
    l3 = tanh(l2.dot(W2))
    #Backpropagation
    d3 = y - l3 #Equation 61
    #Equation 63
    #since log sig derived is x(1 - x)
    d2 = np.multiply(d3.dot(W2.T), vectorizedTanhDeriv(l2))
    #Remove delta for the bias term
    d2 = d2[:, 1:]

    #Equation 59
    W2gradient = np.dot(l2.T, d3)
    W1gradient = np.dot(l1.T, d2)

    #Equation 53
    W2 += learn_rate * W2gradient
    W1 += learn_rate * W1gradient

W1 = np.random.random((3, 10))
W2 = np.random.random((11, 1))

X = []
Y = []
for i in range(0, 1000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    X.append([x1,x2])
    Y.append(np.sign(x1*x2))

Xtest = []
Ytest = []
for i in range(0, 100):
    xt = random.uniform(-1, 1)
    yt = random.uniform(-1, 1)
    Xtest.append([xt,yt])
    Ytest.append(np.sign(xt*yt))

Xpp = []
Ypp = []
for i in range(0, 1000):
    xt = random.uniform(-1, 1)
    yt = random.uniform(-1, 1)
    Xpp.append([xt,yt])
    Ypp.append(np.sign(xt*yt))

epochs = 200
error = []
procerror = []

for x in range(epochs):
    if x % 5 == 0:
        #print "Epochs: %d" % x
        errorval = 0
        procerrorval = 0
        for i in range(len(Xtest)):
        #Forward pass to test values now
            W1temp = W1
            W2temp = W2
            l1 = np.matrix(Xtest[i], dtype = 'f')
            l1 = np.c_[np.ones(1), l1]
            l2 = np.c_[np.ones(1), vectorizedTanh(l1.dot(W1temp))]
            l3 = tanh(l2.dot(W2temp))
            l3cpy = l3
            l3proc = l3cpy
            
            #print "%d XOR %d = %.7f" % (l1[0,1], l1[0,2], l3[0,0])
            procerrorval += np.abs(Ytest[i] - output_process(l3proc[0,0]))
            errorval += np.abs(Ytest[i] - l3cpy[0,0])
        error.append(errorval)
        procerror.append(procerrorval)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        backprop(x, y, W1, W2)



xvals = range(len(error))

plt.figure(1)
plt.subplot(211)
plt.plot(xvals, error)
plt.xlabel('Epochs', fontsize='smaller')
plt.ylabel('Average error rate', fontsize='smaller')
plt.title('Average Error Rate at n Epochs with Unprocessed Output')

plt.subplot(212)
plt.plot(xvals, procerror)
plt.xlabel('Epochs',  fontsize='smaller')
plt.ylabel('Average error rate', fontsize='smaller')
plt.title('Average Error Rate at n Epochs with Processed Output')
plt.subplots_adjust(hspace=0.5)
plt.savefig("resextra.png")

plt.show()

result = []
for i in range(0, 1000):
    W1temp = W1
    W2temp = W2
    l1 = np.matrix(Xpp[i], dtype = 'f')
    l1 = np.c_[np.ones(1), l1]
    l2 = np.c_[np.ones(1), vectorizedTanh(l1.dot(W1temp))]
    l3 = tanh(l2.dot(W2temp))
    result.append(output_process(l3[0,0]))
    print(Ypp[i]-l3[0,0])

xfile = open("points.txt", 'w')
pfile = open("vals.txt", 'w')
for i in range(0, 1000):
    x = str(Xpp[i])[1:-1]
    xfile.write(x)
    xfile.write('\n')
    pfile.write(str(result[i]))
    pfile.write('\n')

xfile.close()
pfile.close()