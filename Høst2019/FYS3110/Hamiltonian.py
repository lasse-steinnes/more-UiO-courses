### Task 3.8H
import numpy as np

## Making objects for
L = 25
g = 1
V = 5

# Setting up the hamiltonian matrix

H = np.zeros((L,L))

# Setting V in the matrix
H[0,0] = V

for i in range(0,L-1):
    H[i+1,i] = -g
    H[i,i+1] = -g
print(H)

# Making the position matrix
X = np.zeros((L,L))

for i in range(0,L-1):
    
