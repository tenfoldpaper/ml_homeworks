import numpy as np
import matplotlib.pyplot as plt
f = open('results.txt', 'w')
# sigmoid
def log_sig(a):
    return 1/(1+np.exp(-a))

def log_sig_derived(a):
    return log_sig(a)*(1-log_sig(a))

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

W1 = np.random.random((3, 2))
W2 = np.random.random((3, 1))

X = [[0,0],[1,0],[0,1],[1,1]]
Y = [0, 1, 1, 0]
epochs = 10000
error = []
for x in range(epochs):
    if x % 10 == 0:
        #print "Epochs: %d" % x
        errorval = 0
        for i in range(4):
        #Forward pass to test values now
            W1temp = W1
            W2temp = W2
            l1 = np.matrix(X[i], dtype = 'f')
            l1 = np.c_[np.ones(1), l1]
            l2 = np.c_[np.ones(1), vectorizedSigmoid(l1.dot(W1temp))]
            l3 = log_sig(l2.dot(W2temp))
            l3cpy = l3
            #print "%d XOR %d = %.7f" % (l1[0,1], l1[0,2], l3[0,0])
            errorval += np.abs(Y[i] - l3cpy[0,0])
        error.append(errorval/2)
    for i in range(4):
        x = X[i]
        y = Y[i]
        backprop(x, y, W1, W2)
    

for i in range(4):
    #Forward pass to test values now
    l1 = np.matrix(X[i], dtype = 'f')
    l1 = np.c_[np.ones(1), l1]
    l2 = np.c_[np.ones(1), vectorizedSigmoid(l1.dot(W1))]
    l3 = log_sig(l2.dot(W2))
    print "%d XOR %d = %.7f" % (l1[0,1], l1[0,2], l3[0,0])

xvals = range(1, 1001)
plt.plot(xvals, error)
plt.xlabel('epochs * 10')
plt.ylabel('Average error % ')
plt.title('Error rate after n*10 epochs')
plt.savefig("res.png")
plt.show()
