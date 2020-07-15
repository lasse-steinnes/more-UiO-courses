## program to solve the equations of motion (eom)
##  using Euler Chromer
# all neeed is a initial conditions

import numpy as np
import matplotlib.pyplot as plt

def a_(r):
    g = 9.81 # m/s^2
    global p; p = 1.4
    global m; m = 1
    return 1/2*(p**2/(m**2*r**3) - g)


def Euler_Chromer(a_,v0,r0,N,t):
    '''
    Euler chromer return arrays for acceleration, velocity, position
    and time.
    Arguments:
    t: time span in seconds
    N: Gives number of time intervals (N-1)
    v0: Initial radial velocity
    r0: Initial radial position
    '''
    global dt; dt = t/N
    r = np.zeros(N); v = np.zeros(N); a_n = np.zeros(N); t = np.zeros(N)
    r[0] = r0; v[0] = v0
    index = N
    for i in range(N-1):
        a_n[i] = a_(r[i])
        v[i+1] = v[i] + a_n[i]*dt
        r[i+1] = r[i] + v[i+1]*dt
        t[i+1] = t[i] + dt
        if r[i+1] < 0.01 and r[i+1] > -0.01:
            break
            index = i
    return index, a_n, v, r, t

v0 = 1 # m/s
r0 = 1 # m
t = 10**2
N = 10**3

j,a,v,r,t = Euler_Chromer(a_,v0,r0,N,t)

# Plotting the solution in space
t0 = 0
v_theta = lambda r: p/(m*r**2) # let t0 = 0
# solving euler chromer for theta
theta = np.zeros(N)

for i in range(j-1):
    theta[i+1] = theta[i] + v_theta(r[i+1])*dt

x = r*np.cos(theta)
y = r*np.sin(theta)
plt.plot(x,y)
plt.show()
