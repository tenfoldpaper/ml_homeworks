import numpy as np

# Initialize dataset, 
data = [] # Append the 1000 data into this array

## Here, include the dataset from one of the previous homeworks (pipeline & text read? OR Copy paste; whatever.)

# Functionalize the PCA equation, and set its resulting k feature vectors (u) as the return value.  
# pca(data, k) -> mat(m x k)

k = 1 # Number of features we want to extract 
pcamat = pca(data, k)

fvecmat = []
for (i in range (0, 1000)):
    fvecmat.append(np.append([np.transpose(pcamat) * data[i], 1])

# PCA's u vectors are the "feature" vectors; so, u'j * xi = phi_j (i) (ith element of phi_j)

# The labels transformed into binary vectors; 1 at the end for bias 
z1 = np.array([1,0,0,0,0,0,0,0,0,0]) # first 100
z2 = np.array([0,1,0,0,0,0,0,0,0,0]) # second 100
z3 = np.array([0,0,1,0,0,0,0,0,0,0]) # third 100 ... 
z4 = np.array([0,0,0,1,0,0,0,0,0,0])
z5 = np.array([0,0,0,0,1,0,0,0,0,0])
z6 = np.array([0,0,0,0,0,1,0,0,0,0])
z7 = np.array([0,0,0,0,0,0,1,0,0,0])
z8 = np.array([0,0,0,0,0,0,0,1,0,0])
z9 = np.array([0,0,0,0,0,0,0,0,1,0])
z0 = np.array([0,0,0,0,0,0,0,0,0,1])
ztot = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z0]

# Key problem of the assignment: Figure out how to get Phi. 

# Calculate fv(xi) for i = [1, N] -> m-dimensional vector. Name the result fv1 ... fvN.  
# then phi_1 = a vector comprised of the first values of the above calculation
# and so on, the last vector will be phi_m
# Phi is then made from this 
# Make an append loop. 

#Calculating Z. 
# 1st 100 rows = z1,
# 2nd 100 rows = z2, 
# 3rd 100 rows = z3, ... 

Z = z1

for (i in range(0, 99)):
    Z = np.append([Z, z1], axis = 0)

for (j in range (1, 10)):
    for (i in range (0, 100)):
        Z = np.append([Z, ztot[j]], axis = 0)

# Calculate Wopt.

SEkTrain = 0

for i in range (0, 1000):
    SEkTrain += pow(Wopt * fvecmat[i] - Z[i], 2)

SEkTrain /= 1000

MISSTrain = 0

for i in range (0, 1000):
    if(max(Wopt * fvecmat[i]) != Z[i]):
        MISSTrain += 1

