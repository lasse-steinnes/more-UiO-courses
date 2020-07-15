### Final exam 2020: Program for task 2: Scattering ###

##import packages
import matplotlib.pyplot as plt
import numpy as np

### Define cross section for free and bound electron
### Define constants
pi_fac = 2*np.pi
r0 = 2.8179*10**(-15) # [m]
omega0 =  2.0*10**16 # [1/s]
gamma = 6.0*10**15 # [1/s]

def Thompson(omega,r0):
    return np.zeros(len(omega)) + 8/3*np.pi*r0**2

def Rayleigh(omega,omega0,gamma,r0):
    C = 8/3*np.pi*r0**2
    func = omega**4/((omega**2 - omega0**2)**2 + omega**2*gamma**2)
    return C*func

# Define array of omega and plot
omega1 = np.linspace(2.0*10**15,5*10**16,1000)
omega2 = np.linspace(405,790,1000)*10**12

#print(omega1)
#print(omega2)


### Figure 3
plt.figure(figsize = (8,6))
plt.plot(omega1/pi_fac,Thompson(omega1,r0),'--', label = 'Free electron')
plt.plot(omega1/pi_fac,Rayleigh(omega1,omega0,gamma,r0), label = 'Bound electron')
plt.legend()
plt.title('Figure 3: $\sigma$ as a function of $f$')
plt.xlabel('Frequency, f [$s^{-1}$]')
plt.ylabel('Scattering cross section, $\sigma$ [$m^{2}$]')
plt.show()


### Figure 5
plt.figure(figsize = (8,6))
plt.plot(omega2/pi_fac,Rayleigh(omega2,omega0,gamma,r0), label = 'Bound electron')
plt.legend()
plt.title('Figure 5: $\sigma$ as a function of $f$ for visible light')
plt.xlabel('Frequency, f [$s^{-1}$]')
plt.ylabel('Scattering cross section, $\sigma$ [$m^{2}$]')
plt.show()
