#### Program for parametrizised solution to the brachistome problem
#### Exercise 2d) Problem set 5

import numpy as np
import matplotlib.pyplot as plt

# Creating the array
k = 1
N = 4
theta = np.linspace(0,2*np.pi*N,1000)
x = 1/2*k**2*(theta - np.sin(theta))
y = 1/2*k**2*(np.cos(theta) - 1)



plt.plot(x,y)
plt.title('Solution to the brachistome problem \n Shortest path in the x-y plane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
