import random
import math
import numpy as np
from numpy import linalg as LA
random.seed()
random.randint(0, 10)


# sigmoid
def log_sig(a):
    return 1/(1+math.exp(-a))

def log_sig_derived(a):
    return log_sig(a)*(1-log_sig(a))

# eq 47
def forward_pass_hidden_layer(W, x, n):#x = xm-1 in notes, n = length of the vector
    #Applies sigmoid to the result
    res = (W*x)
    temp = x.copy()

    for i in range(0, n-1):
        temp[i] = log_sig(res[i])
    temp[n-1] = 1
    return temp

# eq 48
def forward_pass_output(W, x):
    return (W*x)

# eq 49
def loss_function(test_label, train_labe):
    return pow((train_label-test_label),2)

# eq 50
def emp_risk_function(test_labelset, train_labelset):
    res = 0
    for i in range (0, len(test_labelset)):
        res += loss_function(test_labelset[i], train_labelset[i])
    return res/(len(test_labelset)+1)

# eq 53
def update_model(model, stepsize, risk_gradient):
    return model - (stepsize*risk_gradient)

# eq 55: potential of unit x
def x_potential(W, x_ms1, layer_size): #ms1 stands for m subtract 1
    W1 = W[:, 0:layer_size-1]
    x1 = x_ms1[0:layer_size-1]
    res = W1*x1
    return res

# eq 59

def loss_derived_wrt_wmij(d, x):
     return d*x

# eq 61
def delta_output(yhat, y):
    return 2*(yhat - y)

def delta_hlayer_individual(pot_at_mi, delta_next_layer, w_next_layer_toI, next_layer_size):
    siggy = log_sig_derived(pot_at_mi)
    sumRes = 0
    if(next_layer_size == 1): next_layer_size += 1 #this is for the output layer, where there is no bias
    for l in range(0, next_layer_size - 1): #next_layer_size - 1 to ignore the last component of each layer, the bias
        sumRes += (delta_next_layer * w_next_layer_toI[l])
    return siggy * sumRes
# eq 63
def delta_hlayer(pot_vector_m, delta_next_layer, w_next_layer, next_layer_size, hidden_layer_size):
    res = []
    for i in range(0, hidden_layer_size - 1): #ignoring bias again
        tempRes = delta_hlayer_individual(pot_vector_m[[i],:], delta_next_layer, w_next_layer[:,[i]], next_layer_size)
        res.append(tempRes)
    return res
#step size
step_size = 0.1

#data, data, bias
input_data = [[1.0,1.0,1.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[0.0,0.0,1.0]]
input_vectors =np.matrix(input_data).transpose()
#retrieving column vector: np.matrix(input_data).transpose()[:, [0]]

#labels
output_label = [0,1,1,0]

# MLP Structure
bias = 1
n = 0


#Weight matrices
W1 = np.matrix(np.random.rand(2,3))
W2 = np.matrix(np.random.rand(1,3))
for notusedbefore in range(0, 10000):
    for n in range(0, 4):
        input_layer = input_vectors[:, [n]]
        #calculating activation vector x1
        hidden_layer = forward_pass_hidden_layer(W1, input_layer, len(input_layer))

        hidden_layer_potential = x_potential(W1, input_layer, len(input_layer))
        output_layer = forward_pass_output(W2, hidden_layer)
        output_layer_potential = x_potential(W2, hidden_layer, len(hidden_layer))

        delta_k = delta_output(output_layer, output_label[n])
        #since delta_k is a single value, we change that to scalar.
        delta_ks = np.asscalar(delta_k)

        bp_deriv1 = loss_derived_wrt_wmij(delta_ks, hidden_layer).transpose()

        delta_h = delta_hlayer(hidden_layer_potential, delta_ks, W2, len(output_layer), len(hidden_layer))

        bp_deriv2 = np.array([])
        for i in range(0, len(delta_h)):
            for j in range(0, len(input_layer)):
                bp_deriv2 = np.append(bp_deriv2, np.array([delta_h[i] * W1[i, j]]))

        bp_deriv2.shape = (2,3)
        bp_deriv2 = np.matrix(bp_deriv2)

        w2pre = LA.norm(W2)
        w1pre = LA.norm(W1)
        W2 = update_model(W2, step_size, bp_deriv1)
        W1 = update_model(W1, step_size, bp_deriv2)
        print(delta_k)
'''
#To do:
1. Write the weight update function for one neuron. {DONE} See if it can be reliably computed with different variables (eqn 53)
2. Weight update function should use the BP functions already defined
3. Numpy nightmare
4. Bias is always 1. it's the weights that change.
5. Step size manipulation? Usually larger is used in the beginning, and smaller at the later epochs

a. There are 3 deltas, 2 hidden and 1 output
b. There are 6 derived values, permutation of UNIT TO UNIT combination between each layer:
    input layer has 2 units (excl bias).
    hidden layer has 2 units (excl bias).
    output layer has 1 unit.
    2x2 + 2x1 = 6.
    Matter of reshuffling the equations to calculate only the ones that matter, i.e. anything that's NOT weight.

Step 0: Initialize weight matrix W1 (3x3) and W2 (3x1) with small random values
Step 1: Start backprop on existing set.
    For i = 0 to and including bias (if it exists), repeat:
        1.1: forward pass: eq. 47, 48, 55 -- potential error source
        1.2: Compute delta K: eq. 60 & 61
        1.3: Compute delta n != K: eq. 63
        1.4: Put the values into the corresponding matrix position (fill up the row first)
Step 2: Run eq. 53 with the new weight matrices and the old weight matrices
Step 3. Repeat 1 and 2 until threshold has been reached (found in page 81) or n number of repeats have been completed
    # Changing the step size at each iteration might be wise.
Step 4. Return the newest weight vector.
'''
