# Program to describe madelung constant #
import numpy as np
import matplotlib.pyplot as plt

# Plotting the madelung constant

def madelung(N):
    """
    Takes N unit cells in the crystal as argument.
    returns
    """
    N = 2*N
    alfa = np.zeros(N)
    alfa[0] = 1
    for i in range(1,N):
        alfa[i] = (-1)**(i)/(i+1)
    cumsum = np.cumsum(alfa)
    atom_i = np.linspace(1,N,N)
    logapprox = np.zeros(N)
    for i in range(N):
        logapprox[i] = np.log(2)
    plt.plot(atom_i,cumsum/N,'.-')
    plt.plot(atom_i,logapprox,'--')
    plt.xlabel('atom index')
    plt.ylabel('Madelung (unitless)')
    plt.show()
    return atom_i, cumsum

#madelung(50)

# Plot of corresponding cohesive energy evolution
def U_tot(func,N,R0):
    """
    Takes the function madelungs and number of unit cells as argument
    and returns cohesive energy evolution
    """
    eps0 = 55.26349406/((10**9)*10**(-15)) #eV metres
    q_squared_SI = 1/(4*np.pi*eps0)
    rho = 0.32*10**(-10) #Ångstrøm --> meter
    atom_i, alfa = func(N)
    N = 2*N
    Utot = - (N*alfa*q_squared_SI)/R0*(1-rho/R0)
    plt.plot(atom_i,Utot/N)
    plt.ylabel('eV/N')
    plt.xlabel('atom index')
    plt.show()
    return Utot/N

#3R0 = 2.820*10**(-10) # Ångstrøm --> meter
#U_tot(madelung,50,R0)

# Plot of equlibrium separation of
# between Na+ and Cl- as function of chrystal size

def eq_separation(func1,N,R0):
    eps0 = 55.26349406/((10**9)*10**(-15)) #eV metres
    q_squared_SI = 1/(4*np.pi*eps0)
    rho = 0.32*10**(-10) #Ångstrøm --> meter
    atom_i, alfa = func1(N)
    N = 2*N
    Utot = -(N*alfa*q_squared_SI)/R0*(1-rho/R0)
    plt.plot()
    plt.show()

R = np.linspace(1,6,100)*10**(-10)
