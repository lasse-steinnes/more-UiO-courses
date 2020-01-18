# tries subset of matrix
import numpy as np

N = 5
x = np.linspace(0,N,N+1)
y = np.linspace(0,N,N+1)
X,Y = np.meshgrid(x,y)
red = np.zeros((6,3))
print(X)
for i in range(len(X[1][:])-1):
 red[i,:] = X[i,::2]
 print(i)
print(red)

print(X[1,:])
print(np.shape(X))
print(np.shape(red))

N = len(terrain[0][::4]) # lengde kolonner
n = len(terrain[::4][0]) # lengde antall rad
#NN = len(terrain[0][:]) # number of columns total
nn = len(terrain[:][0]) # number of rows total
#print(n,N)
print(np.shape(terrain))
reduced  = np.zeros((n,N))

for i in range(nn):
    reduced[i,:] = terrain[i][::4]

for j in range(N):
    reduced[:,j] = terrain[::4][j]

print(reduced)
print(np.shape(reduced))
