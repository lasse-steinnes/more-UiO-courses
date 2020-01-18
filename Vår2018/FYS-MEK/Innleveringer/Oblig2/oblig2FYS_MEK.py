### Løsning av differensiallikningen
### Skal finne posisjonen

# h) Skriv opp ligningene for å løse

# i) Løs med init, plot for de første 10 sek.
# Skal helst bruke vektoriserte beregninger.
# Lager en høyreside:

import numpy as np


k = 200 # N/m
L0 = 1 # m
m = 0.1 #kg
g = 9.81 # tyngdeakselerasjon # retning er med i ligningen

def a(x,y,t):
    a_x = -k*x/m*(1 - L0/(np.sqrt((x**2) + (y**2))))
    a_y = -k*y/m*(1 - L0/(np.sqrt((x**2) + (y**2)))) - g
    return [a_x,a_y]


n = 10000            # tidssteg
T = 10              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a_arr = np.zeros((n,2))     # Array for akselerasjon
v = np.zeros((n,2))     # Array for fart
r = np.zeros((n,2))     # Array for posisjon
t = np.zeros(n)     # Tidsarray

# Initialverdiene
r[0,0] =  0.5
r[0,1] = - 0.866
# Eulers Cromers metode:

for i in range(0,n-1):
    for j in range(2):
        a_arr[i,j] = a(r[i,0],r[i,1],t[i])[j]
        v[i+1,j] = v[i,j] + dt*a_arr[i,j]
        r[i+1,j] = r[i,j] + dt*v[i+1,j]
        t[i+1] = t[i] + dt

# Bra :D

# Skal nå plotte
import matplotlib.pyplot as plt
plt.plot(r[:,0],r[:,1], linewidth = 0.5)
plt.plot([0,r[0,0]],[0,r[0,1]],'-or')
plt.title('Banen for en pendel med fjærkraft')
plt.xlabel('x(t) [m]')
plt.ylabel('y(t) [m]')
plt.axis('equal')
#plt.savefig('oblig1graf1.png')
plt.show()


# Oppgave j
# med delta t 10000

# k: med delta t = 1000, ok
# med delta t = 100, overflow, 1e121
# l ) 1e301 Overflow due to double scalars.

# Oppgave m) Modifisere programmet for nye betingelser
import numpy as np

k = 200 # N/m
L0 = 1 # m
m = 0.1 #kg
g = 9.81 # tyngdeakselerasjon # retning er med i ligningen
# antar at a0  er 0

def a(x,y,t):
    if np.sqrt((x**2) + (y**2)) >= L0:
        a_x = -k*x/m*(1 - L0/(np.sqrt((x**2) + (y**2))))
        a_y = -k*y/m*(1 - L0/(np.sqrt((x**2) + (y**2)))) - g
    else:
        a_x = 0
        a_y = - g

    return [a_x,a_y]


n = 10000            # tidssteg
T = 10              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a_arr = np.zeros((n,2))     # Array for akselerasjon
v = np.zeros((n,2))     # Array for fart
r = np.zeros((n,2))     # Array for posisjon
t = np.zeros(n)     # Tidsarray

# Initialverdiene
r[0,0] = 1
r[0,1] = -0.8
v[0,0] = 6.0
# Eulers Cromers metode:

for i in range(0,n-1):
    for j in range(2):
        a_arr[i,j] = a(r[i,0],r[i,1],t[i])[j]
        v[i+1,j] = v[i,j] + dt*a_arr[i,j]
        r[i+1,j] = r[i,j] + dt*v[i+1,j]
        t[i+1] = t[i] + dt

# Bra :D

# Skal nå plotte
import matplotlib.pyplot as plt
plt.plot(r[:,0],r[:,1], linewidth = 0.3)
plt.plot([0,r[0,0]],[0,r[0,1]],'-or', linewidth = 0.3 )
plt.title('Modifisert modell for pendelbanen')
plt.xlabel('x(t) [m]')
plt.ylabel('y(t) [m]')
plt.axis('equal')
#plt.savefig('oblig1graf1.png')
plt.show()
