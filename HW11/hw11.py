import random
import math
import numpy as np

random.seed()
random.randint(0, 10)

# sigmoid
def log_sig(a):
    return 1/(1+math.exp(-a))

def log_sig_derived(a):
    return log_sig(a)*(1-log_sig(a))

# eq 49
def loss_function(test_label, train_labe):
    return pow((train_label-test_label),2)

# eq 50
def emp_risk_function(test_labelset, train_labelset):
    res = 0
    for i in range (0, len(test_labelset)):
        res += loss_function(test_labelset[i], train_labelset[i])
    return res/(len(test_labelset)+1)

# eq 55: potential of unit x 

def x_potential(w_m_i, x_ms1, layer_size): #ms1 stands for m subtract 1 
    res = 0
    for j in range(0, layer_size):
        res += w_m_i[j] * x_ms1[j]
    return res 

# eq 59

 def loss_derived_wrt_wmij(d, x):
     return d*x

# eq 61
def delta_output(yhat, y):
    return 2*(yhat - y)

# eq 63
def delta_hlayer(pot_a_atI, delta_next_layer, w_next_layer_toI, next_layer_size):
    res = 0 
    siggy = log_sig_derived(pot_a_atI)
    for i in range (0, next_layer_size):
        res += (delta_next_layer*w_next_layer_toI)
    return siggy*res

#step size 
step_size = 0.001

#data, data, label 
input_data = [[1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]

# MLP Structure
input_layer = [a, b, b0]
hidden_layer = [x1, x2, b1]
output_layer = yhat

#Weight matrices
W1 = np.random.rand(3,3)
W2 = np.random.rand(3,1)


'''
#To do: 
1. Write the weight update function for one neuron. See if it can be reliably computed with different variables (eqn 53)
2. Weight update function should use the BP functions already defined
3. Numpy nightmare
4. Try both settings for bias: either always 1, or only the first one is 1 and update the rest in backprop.
5. Step size manipulation? Usually larger is used in the beginning, and smaller at the later epochs 

Step 0: Initialize weight matrix W1 (3x3) and W2 (3x1) with small random values 
Step 1: Start backprop on existing set.
    For i = 0 to and including bias (if it exists), repeat:
        1.1: forward pass: eq. 55
        1.2: Compute delta K: eq. 60 & 61
        1.3: Compute delta n != K: eq. 63
        1.4: Put the values into the corresponding matrix position (fill up the row first)
Step 2: Run eq. 53 with the new weight matrices and the old weight matrices
Step 3. Repeat 1 and 2 until threshold has been reached (found in page 81) or n number of repeats have been completed
    # Changing the step size at each iteration might be wise. 
Step 4. Return the newest weight vector. 
'''

