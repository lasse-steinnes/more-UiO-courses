import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0

### oppgave 1
def E(x, y):
    r1 = np.array([x-1.0, y-3.0])
    r2 = np.array([x-2.0, y-4.0])
    r1norm = np.sqrt((x-1.0)**2 + (y-3.0)**2)
    r2norm = np.sqrt((x-2.0)**2 + (y-4.0)**2)

    E1 = 0.45/(4*np.pi*epsilon_0*r1norm**3)*r1
    E2 = -0.45/(4*np.pi*epsilon_0*r2norm**3)*r2
    return E1 + E2

x = np.linspace(0, 5.6, 20)
y = np.linspace(0, 4.8, 20)

X, Y = np.meshgrid(x, y)

Ex, Ey = E(X, Y)

plt.figure(figsize=(4, 3))
plt.quiver(X, Y, Ex, Ey)
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.tight_layout()
plt.show()
####

###
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0

def E(r):
    r1 = r - np.array([0.01, 0.03])
    r2 = r - np.array([0.02, 0.04])

    E1 = 0.45/(4*np.pi*epsilon_0*np.sum(r1**2)**(3/2))*r1
    E2 = -0.45/(4*np.pi*epsilon_0*np.sum(r2**2)**(3/2))*r2
    return E1 + E2

Q = -2e-8
m = 1.3

dt = 1e-7
T = 0.01
Nt = int(T/dt)

r = np.zeros((Nt, 2))
v = np.zeros((Nt, 2))
v[0,1] = 100.0

for i in range(Nt-1):
    a = Q*E(r[i])/m
    v[i+1] = v[i] + a * dt
    r[i+1] = r[i] + v[i+1] * dt

plt.figure(figsize=(4, 3))
plt.plot(r[:,0], r[:,1], label="Partikkelens bane")
plt.plot([0.01], [0.03], "ro", label="$Q_1$")
plt.plot([0.02], [0.04], "bo", label="$Q_2$")
plt.legend()
plt.xlim([-0.05, 0.05])
plt.ylim([-0.01, 0.1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.show()
