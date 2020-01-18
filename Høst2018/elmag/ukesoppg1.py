import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
### Oppgave 2
## Skal plotte det elektriske feltet ###
"""
eps = 8.85*10**(-12)  C^2 N^-1 m^-2 permittivitet i vakuum
"""

def E_felt(q0,r,r0): # alle variable er enhetsløse r/a, a er karakteristisk avstand
    dr = r - r0   # r0: pos av q0
    rnorm = np.sqrt(dr.dot(dr))
    return q0*dr/rnorm**3 # E/E0, der E0 = er i avstand a (kommer med i dr?)

## feltet som funksjon av radius
r = np.linspace(-2,2,100)
E = np.zeros(r.shape)
for i in range(len(r)):
    ri = np.array(r[i])
    r0 = np.array([0])
    E[i] = E_felt(1.0,ri,r0)

plt.figure(figsize = (6,3))

ax1 = plt.subplot(1,2,1)
ax1.set_xlabel('r')
ax1.set_ylabel('E/E0')
ax1.plot(r,E)

ax2 = plt.subplot(1,2,2)
ax2.set_xlabel('r')
ax2.set_ylabel('E/E0')
ax2.loglog(r,E) # logaritmisk på begge aksene

plt.tight_layout()
plt.show()

# b) Et plot som viser feltet i xy-planet:
# Et plot som viser feltlinjene(strømlinjene) i xy-planet. plt.quiver
# initialverdier
R0 = np.array([0.0,0.0])
q0 = 1.0

# Lager et grid
x = np.linspace(-5,5,21)
y = np.linspace(-5,5,21)
X,Y = np.meshgrid(x,y)
Ex = X.copy()
Ex[:] = 0  # lager tomme array for elektriske feltet med samme dimensjoner
Ey = Y.copy()
Ey[:] = 0

for i in range(len(X.flat)):     # når man flater ut blir indeksering 0,1,2...
    R = np.array([X.flat[i],Y.flat[i]])    # husk at du må bruke riktig klamme
    Ex.flat[i],Ey.flat[i] = E_felt(q0,R,R0)      # lager feltet komponentvis
                                                # merk hvordan array kalles på som instanser(tilfeller)

plt.figure(figsize = (6,3))

ax1 = plt.subplot(1,2,1)
ax1.quiver(X,Y,Ex,Ey) # Kobler xy posisjoner til komponentvis ex,ey
ax1.axis('equal')
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2 = plt.subplot(1,2,2)
ax2.streamplot(x,y, Ex, Ey,linewidth=0.75,density=1,arrowstyle='->',arrowsize=0.5)
ax2.axis('equal')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.tight_layout()
plt.show()

# Oppgave 2.c
# skal plotte feltet i en dipol langs x-aksen # Tilsvarende x**3
a = 1.0
def E_dipol(x):
    return 1.0/(x-a)**2 - 1.0/(x+a)**2

x1 = np.linspace(1.05,5,100)    # for å slippe a dele på
x2 = np.linspace(1.05,1000,100) # for å teste tilfellet der x >> a

fig = plt.figure(figsize = (6,3))

ax1 = plt.subplot(1,2,1)
ax1.plot(x1,E_dipol(x1))

ax2 = plt.subplot(1,2,2)
ax2.loglog(x2,E_dipol(x2))
ax2.loglog(x2,1/x2**2)
ax2.loglog(x2,1/x2**3)

plt.show()


# om begge ladningene positive: Tilsvarende x**2
a = 1.0
def E_dipol(x):
    return 1.0/(x-a)**2 + 1.0/(x+a)**2

x1 = np.linspace(1.05,5,100)    # for å slippe a dele på
x2 = np.linspace(1.05,1000,100) # for å teste tilfellet der x >> a

fig = plt.figure(figsize = (6,3))

ax1 = plt.subplot(1,2,1)
ax1.plot(x1,E_dipol(x1))

ax2 = plt.subplot(1,2,2)
ax2.loglog(x2,E_dipol(x2))
ax2.loglog(x2,1/x2**2)
ax2.loglog(x2,1/x2**3)

plt.show()

# oppgave e) Quadropol: 1/x^4
a = 1.0
def E_dipol(x):
    return 1.0/(x-a)**2 + 1.0/(x+a)**2 - 2.0/(x**2+1)

x1 = np.linspace(1.05,5,100)    # for å slippe a dele på
x2 = np.linspace(1.05,1000,100) # for å teste tilfellet der x >> a

fig = plt.figure(figsize = (6,3))

ax1 = plt.subplot(1,2,1)
ax1.plot(x1,E_dipol(x1))

ax2 = plt.subplot(1,2,2)
ax2.loglog(x2,E_dipol(x2))
ax2.loglog(x2,1/x2**2)
ax2.loglog(x2,1/x2**3)
ax2.loglog(x2,1/x2**4)
plt.show()
#######
