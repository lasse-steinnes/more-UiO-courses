import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split

from imageio import imread

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

import scipy

def load_terrain(imname):
# Load the terrain
    terrain = imread('{:}.tif'.format(imname))
#    terraindata = scipy.misc.imread('{:}.tif'.format(imname))
# Show the terrain
    plt.figure()
    plt.title('Terrain over Norway')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
# return the terrain for plotting and the data using scipy
    return terrain

terrain = load_terrain('yellowstone3')
N = len(terrain[0,::4]) # lengde kolonner
n = len(terrain[::4,0]) # lengde antall rad
NN = len(terrain[0,:]) # number of columns total
nn = len(terrain[:,0]) # number of rows total
#print(n,N)
#print(np.shape(terrain))
reduced  = np.zeros((nn,N))


for i in range(nn):
        reduced[i,:] = terrain[i,::4]

reduced2 = np.zeros((n,N))

for j in range(N):
    reduced2[:,j] = reduced[::4,j]

z = reduced2.flatten()
# creating arrays for x and y
x_range = np.arange(1,n+1)
y_range = np.arange(1,N+1)
X,Y = np.meshgrid(x_range,y_range)
x = X.flatten();y = Y.flatten()

print(np.shape(terrain))
#print(reduced)
#print(z)
#print(x)
#print(y)
#print(len(x),len(y),len(z))
#print('shape2',np.shape(reduced2))
#print(np.shape(terrain))
#print(np.shape(reduced))
