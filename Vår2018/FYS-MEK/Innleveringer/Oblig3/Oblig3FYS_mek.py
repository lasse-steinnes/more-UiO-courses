## Obligatorisk innlevering 3 FYS-MEK ##

# e) Skal plotte horisontal fjærkraft som funksjon av x-pos.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

#setter globale parametere
l0 = 0.5 # m
h = 0.3 # m
k = 500 # N/m
m = 5 # kg
g = 9.81 # kg
x0 =  0.4 # m


def F_x(x):
    return -k*x*(1 - l0/np.sqrt(x**2 +h**2))

x = np.linspace(-0.75,0.75,1000)

# plotter
plt.plot(x, F_x(x))
plt.xlabel("x [m]")
plt.ylabel('Fx [N]')
plt.show()

# f)
# Vi har at akselerasjonen er
def a_x(x):
    return -k*x/m*(1 - l0/np.sqrt(x**2 + h**2))


n = 10000            # tidssteg
T = 10              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a_arr = np.zeros(n)     # Array for akselerasjon
v = np.zeros(n)     # Array for fart
x = np.zeros(n)     # Array for x posisjon
t = np.zeros(n)     # Tidsarray

# Initialverdiene
x[0] = 0.6
# De andre er 0

# Eulers Cromers metode:

for i in range(0,n-1):
        a_arr[i] = a_x(x[i])
        v[i+1] = v[i] + dt*a_arr[i]
        x[i+1] = x[i] + dt*v[i+1]
        t[i+1] = t[i] + dt

# dermed blir plottet:

plt.figure(figsize = (10,5))

ax1 = plt.subplot(121)
plt.plot(t,v)
plt.xlabel('tid [s]')
plt.ylabel('Hastighet [m/s]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)


ax2 = plt.subplot(122)
plt.plot(t,x)
plt.xlabel('tid [s]')
plt.ylabel('Posisjon [m]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc)
plt.show()


# g) med x0 = 0.65. Bruke energiargumenter til å forklare
# Initialverdiene

x[0] = 0.65
# De andre er 0

# Eulers Cromers metode:

for i in range(0,n-1):
        a_arr[i] = a_x(x[i])
        v[i+1] = v[i] + dt*a_arr[i]
        x[i+1] = x[i] + dt*v[i+1]
        t[i+1] = t[i] + dt

# dermed blir plottet:
plt.figure(figsize = (10,5))

ax1 = plt.subplot(121)
plt.plot(t,v)
plt.xlabel('tid [s]')
plt.ylabel('Hastighet [m/s]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)


ax2 = plt.subplot(122)
plt.plot(t,x)
plt.xlabel('tid [s]')
plt.ylabel('Posisjon [m]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc)
plt.show()

## l) Skal inkludere friksjonen (må huske at denne krafta kan endre retning)
def a_x(x,v):
    if v < 0:
        ax = k/m*(0.05*h-x)*(1-l0/np.sqrt(x**2 + h**2)) + 0.05*g
    else:
        ax =  -k/m*(0.05*h+x)*(1-l0/np.sqrt(x**2 + h**2)) - 0.05*g
    return ax

# Initialverdiene
x[0] = 0.75
# De andre er 0

# Eulers Cromers metode:

for i in range(0,n-1):
        a_arr[i] = a_x(x[i],v[i])
        v[i+1] = v[i] + dt*a_arr[i]
        x[i+1] = x[i] + dt*v[i+1]
        t[i+1] = t[i] + dt

# dermed blir plottet:

plt.figure(figsize = (10,5))

ax1 = plt.subplot(121)
plt.plot(t,v)
plt.xlabel('tid [s]')
plt.ylabel('Hastighet [m/s]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)


ax2 = plt.subplot(122)
plt.plot(t,x)
plt.xlabel('tid [s]')
plt.ylabel('Posisjon [m]')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc)
plt.show()


# m) Skal plotte den kinetiske energien:

def K1(v):
    return 0.5*m*(v**2)

# kinetisk energi i et objekt
kin = K1(v)

#plot
plt.plot(x,kin)
plt.xlabel('posisjon [m]')
plt.ylabel('Kinetisk energi [Nm (J)]')
plt.axis([-0.8,0.8,0,23])
plt.show()

# Oppgave n)
# Skal finne arbeidet det tar å komme til posisjon x = 0.75 vha numerisk integrasjon
# Det tilsvarer å løse integralet int(f_x) dr. fra x0 til x

def F_x2(x):
    return -k*(0.05*h+x)*(1-l0/np.sqrt(x**2 + h**2)) - 0.05*m*g

x1 = np.linspace(0.4,0.75,1000)
W = np.trapz(-F_x2(x1),x1) # Arbeid fra kraft = - Arbeid mot kraft
print('Arbeidet det tar komme fra x0:{:.1f} til x:{:0.2f} er {:.2f} Nm'.format(x0,0.75,W))
"""Arbeidet det tar komme fra x0:0.4 til x:0.75 er 25.11 Nm"""
## o) Skal finne potensialfunksjonen: potensiell energi. Konservativ kraft, så du/dx = -F

import scipy.integrate
U = scipy.integrate.cumtrapz(-F_x2(x),x)


plt.plot(x[:-1],U)
plt.xlabel('Posisjon [m]')
plt.ylabel('Energi [J]')
plt.show()

# Bestem likevektspunktene.



"""
vec = [x,-F_x2(x)]# tilsvarer cumtrapz for scipy, men numpy er alltid oppdatert
U = np.cumsum(vec,dtype = 'float')
"""

########################################
