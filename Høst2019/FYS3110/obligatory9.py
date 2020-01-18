#### Obligatory assignment 9 #####
import numpy as np

E1 = -13.6 # eV

z1 = np.linspace(0.00,2,1000)
z2 = np.linspace(0.001,2,1000)

def E_tr(x,y):
    return E1/(x**6 + y**6)*(-x**8 + 2*x**7 + 1/2*x**6*y**2\
     - 1/2*x**5*y**2 - 1/8*x**3*y**4 + 11/8*x*y**6-1/2*y**8)

E_min = 10**5
z1_min = 0
z2_min = 0

for i in range(len(z1)):
    for j in range(len(z2)):
        x = z1[i] + z2[j]
        y = 2*np.sqrt(z1[i]*z2[j])

        if E_tr(x,y) < E_min:
            E_min = E_tr(x,y)
            z1_min = z1[i]
            z2_min = z2[j]

print('E minimum: {:},z1: {:}, z2: {:} '.format(E_min,z1_min,z2_min))

''' Run trough
>> Lasse$ python obligatory9.py
E minimum: -13.961837981578723,z1: 1.039039039039039, z2: 0.28314114114114114
'''
