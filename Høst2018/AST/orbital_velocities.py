## Del 1b7##
# estimat for banehastighet til planetene
import numpy as np

p = np.array([4.5,12.8,17.9,39.6,50.6,117.0,230.6,367.2])
p = 8760*60*60*p
a = np.array([4,8,10,17,20,35,55,75])
a = 149597870691/1000*a

def vel(a,p):
    return 2*np.pi*a/p

velocities = vel(a,p)
print(velocities)
