import numpy as np
import matplotlib.pyplot as plt
# sigmoid
def log_sig(a):
    return 1/(1+np.exp(-a))

def log_sig_derived(a):
    return log_sig(a)*(1-log_sig(a))

def output_process(x):
    if(x<=0.5):
        return 0
    else:
        return 1

def process_procerror(p):
    if(p>1):
        return 1
    else:
        return p

#Returns a function vectorizedSigmoid which takes np arrays in
#and returns nparrays too
vectorizedSigmoid = np.vectorize(log_sig)

def backprop(x, y, W1, W2, learn_rate = .2):
    #Forward pass
    l1 = np.matrix(x, dtype = 'f')
    l1 = np.c_[np.ones(1), l1] #adding bias

    #Equations 47 and 48
    l2 = np.c_[np.ones(1), vectorizedSigmoid(l1.dot(W1))]
    l3 = log_sig(l2.dot(W2))
    #Backpropagation
    d3 = y - l3 #Equation 61
    #Equation 63
    #since log sig derived is x(1 - x)
    d2 = np.multiply(d3.dot(W2.T), np.multiply(l2, (1-l2)))
    #Remove delta for the bias term
    d2 = d2[:, 1:]

    #Equation 59
    W2gradient = np.dot(l2.T, d3)
    W1gradient = np.dot(l1.T, d2)

    #Equation 53
    W2 += learn_rate * W2gradient
    W1 += learn_rate * W1gradient
'''
W1 = np.random.random((3, 2))
W2 = np.random.random((3, 1))
'''
X = [[0,0],[1,0],[0,1],[1,1]]
Y = [0, 1, 1, 0]
epochs = 10000
error = []
procerror = []
for blah in range(200, 1, -10):
    W1 = np.random.random((3, 2))
    W2 = np.random.random((3, 1))

    for x in range(epochs):
        #print(x)
        if x % 10 == 0:
            #print "Epochs: %d" % x
            errorval = 0
            procerrorval = 0
            for i in range(4):
            #Forward pass to test values now
                W1temp = W1
                W2temp = W2
                l1 = np.matrix(X[i], dtype = 'f')
                l1 = np.c_[np.ones(1), l1]
                l2 = np.c_[np.ones(1), vectorizedSigmoid(l1.dot(W1temp))]
                l3 = log_sig(l2.dot(W2temp))
                l3cpy = l3
                l3proc = l3cpy
                
                #print "%d XOR %d = %.7f" % (l1[0,1], l1[0,2], l3[0,0])
                procerrorval += np.abs(Y[i] - output_process(l3proc[0,0]))
                errorval += np.abs(Y[i] - l3cpy[0,0])
            error.append(errorval/2)
            procerror.append(process_procerror(procerrorval/2))
            
            if (errorval < 0.05 and x != 0):
                print("Step rate {0} at Epoch {1}".format(blah*0.001, x))
                break
        for i in range(4):
            x2 = X[i]
            y = Y[i]
            backprop(x2, y, W1, W2, 0.001*blah)
        
    
'''
for i in range(4):
    #Forward pass to test values now
    l1 = np.matrix(X[i], dtype = 'f')
    l1 = np.c_[np.ones(1), l1]
    l2 = np.c_[np.ones(1), vectorizedSigmoid(l1.dot(W1))]
    l3 = log_sig(l2.dot(W2))
    print ("%d XOR %d = %.7f" % (l1[0,1], l1[0,2], l3[0,0]))

xvals = range(1, 5001)

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
plt.savefig("res.png")

plt.show()
'''