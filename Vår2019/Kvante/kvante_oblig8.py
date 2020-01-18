##### Program for kvanteoblig 8 #####
import numpy as np
import matplotlib.pyplot as plt

# Sender et elektron med ladning q
q = 1.602*10**(-19)

m = 9.1*10**(-31)
v0 = 10.0 # eV
L = 1.5
ha = 6.58*10**(-16) # eV sec


# Om E < V0
def T1(E):
#print((2*m*(v0-E))/ha*L)
    return 1/(1+v0**2/(4*(v0 - E)*E)*(np.sinh(np.sqrt(2*m*(v0-E))/ha*L))**2)

# Om E = V0
def T2(E):
    return 1/(1+m*L**2/(2*ha**2)*v0)

# Om E > V0
def T3(E):
    return 1/(1 + v0**2/(4*E*(E - v0))*(np.sin(np.sqrt(2*m*(E - v0))/ha*L))**2)

#########################################

E_min = np.linspace(0.1,9.9,1000)
E_max = np.linspace(10.01,100,1000)
E_v0 = v0

plt.title('Transmisjon for en planbølge som funksjon av E')
plt.plot(E_min,T1(E_min),label = 'E < v0')
plt.plot(E_v0,T2(E_v0),'o')
plt.plot(E_max,T3(E_max), label = 'E > v0')
plt.xlabel('E [eV] ')
plt.ylabel('Transmisjon')
plt.legend
plt.show()
##############################
