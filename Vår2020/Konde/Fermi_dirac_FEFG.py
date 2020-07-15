### Program to module III: electrons I task 3 ###
import numpy as np
import matplotlib.pyplot as plt

#####  Must have the right constants ####
### From table 1 in the book # using eV
m = 9.1093826*10**(-31) # kg
#m = 0.510998*10**6  # eV/c**2
k_b = 1.380649*10**(-23) # [J/K]
#hbar = 6.58*10**(-16) # eV*s
hbar =  1.054571817*10**(-34) # [J/s]
#e_C =  4.70*10**22 # [cm^-3] electron consentration specific for Li
e_C = 4.70*10**(22-6)  #m^-3
eps_F = (hbar**2)/(2*m)*(3*np.pi**2*e_C)**(2/3)


T_F = eps_F/k_b

print('eps_F:',eps_F)
print('T_F:',T_F)
"""
To avoid overflow, maybe work with quantum mass
"""

#### Creating arrays ####
eps = np.linspace(0.0,4*eps_F,1000)
T = np.array([0.01*T_F, 0.1*T_F,0.5*T_F,1.0*T_F])
print('T_list:',T)

def f(eps,T):
    return 1/(np.exp(T_F/T*(eps/eps_F-1)) + 1)

for i in range(len(T)):
    plt.plot(eps/eps_F,f(eps,T[i]))

plt.xlabel('$\epsilon / \epsilon_F$')
plt.ylabel('$f(\epsilon)$')
plt.show()

########### 3c ###########
# Numerical integration: The easy way

### Need to define density of states


##########################
