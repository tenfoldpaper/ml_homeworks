import numpy as np 
import math

W=np.matrix('1 2 3; 4 5 6; 7 8 9')
x=np.matrix('10;20;30')
wm=np.matrix('1;1;1')
print(W*x + wm)

def log_sig(a):
    return 1/(1+math.exp(-a))
