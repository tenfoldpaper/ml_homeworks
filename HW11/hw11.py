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

step_size = 0.001

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

#data, data, label 
input_data = [[1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]

# MLP Structure
input_layer = [x, y, 1]

hidden_layer = [a, b, 1]

output_layer = 1
