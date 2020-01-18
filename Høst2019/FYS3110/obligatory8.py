#### Task 8.6 ####
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Defining parameters
N = 100
a = 0.01
l = np.linspace(0,N-1,100)
beta = 1 #m*a*alfa/h_red**2

K = lambda l: 2*np.pi*l/(N*a)

#function defining z
# finding roots
initial = np.array([1.5, 5.4, 7.9])
z_r = np.zeros((len(l),len(initial)))

for i in range(len(l)):
        func = lambda z: np.cos(z) + beta*np.sin(z)/z - np.cos(K(l[i])*a)
        z_r[i,:] = fsolve(func,initial)

print('These are the z-roots for l  = (1,2,...100):\{:}'.format(z_r))

E = z_r**2

plt.figure()
plt.title('Task 8.6')
plt.plot(K(l)*a,E[:,0], label = 'E0')
plt.plot(K(l)*a,E[:,1], label = 'E1')
plt.plot(K(l)*a,E[:,2], label = 'E2')
plt.xlabel('Ka')
plt.ylabel('E')
plt.legend()
plt.show()

#### 8.7 (X) ###
En = np.array([1,1,4/3,3/2,8/5,5/3,13/7,2])
n = np.linspace(1,8,8)
plt.figure()
plt.title('Task 8.7')
plt.plot(n,En)
plt.xlabel('number of particles [n]')
plt.ylabel('energy [En/hw]')
plt.show()
