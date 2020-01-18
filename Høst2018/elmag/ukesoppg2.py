### Oppgaver til uke 2
import numpy as np
import matplotlib.pyplot as plt
"""
### Oppgave 2; Programmere dipol og gradient
# a) det oppstår et skalarpotensiale i xy-planet.
# Dette skal visualiseres med konturlinjer, dvs linjer der V = C
def V_dobbel_positiv(x, y):
    return 1.0/np.sqrt((x-1)**2+y**2) + 1.0/np.sqrt((x+1)**2+y**2) # potensialfunksjonen

def V_dipol(x, y):
    return 1.0/np.sqrt((x-1)**2+y**2) - 1.0/np.sqrt((x+1)**2+y**2)

x = np.linspace(-2, 2, 30) # lager x-y array
y = np.linspace(-2, 2, 30)

X, Y = np.meshgrid(x, y)  #fordeler x og y over et grid

Varr = V_dobbel_positiv(X, Y)
plt.figure(figsize=(6,3))
plt.subplot(121)
plt.contourf(X, Y, Varr, 20) # contourf; int er levels; ie. hvor mange konturlinjer som lages
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")


# b) Beregne elektrisk felt ved å ta gradient numerisk, illustrert med vektorplot
plt.subplot(122)
Eypospos, Expospos = np.gradient(-Varr) # merk rekkefølgen her og fortegn!! E = -Del(V)
plt.quiver(X, Y, Expospos, Eypospos, scale=50, width=0.005)
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")
plt.tight_layout()
plt.show()

# oppgave c
# for en dipol
Varr2 = V_dipol(X,Y)

plt.figure(figsize=(6,4))
plt.subplot(221)
plt.contourf(X, Y, Varr2, 20) # contourf; int er levels; ie. hvor mange konturlinjer som lages
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")

plt.subplot(222)
Ey, Ex = np.gradient(-Varr2) # merk rekkefølgen her og fortegn!! E = -Del(V)
plt.quiver(X, Y, Ex, Ey, scale=50, width=0.005)
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")
plt.tight_layout()

# strømlinjer for dipol
plt.subplot(223)
plt.streamplot(X,Y,Ex,Ey)
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")
plt.tight_layout()

plt.subplot(224)
color = 2 * np.log(np.hypot(Expospos, Eypospos))
plt.streamplot(X,Y,Expospos,Eypospos, color = color, cmap = plt.cm.inferno)
plt.colorbar()
plt.xlabel("$x/a$")
plt.ylabel("$y/a$")
plt.tight_layout()
plt.show()
"""

# oppgave 3

z = np.linspace(0,20,100)
a = 2.0

def E(z): # utelater konstanter
    return 1-z/np.sqrt(a**2 + z**2)

plt.plot(z,E(z))
plt.xlabel('z')
plt.ylabel(r'$2\epsilon_0E(z)/rho_s$')
plt.show()







#
