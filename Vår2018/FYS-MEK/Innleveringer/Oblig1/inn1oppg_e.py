
import numpy as np

n = 1000            # tidssteg
T = 10              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a = np.zeros(n)     # Array for akselerasjon
v = np.zeros(n)     # Array for fart
x = np.zeros(n)     # Array for posisjon
t = np.zeros(n)     # Tidsarray
c = 7/1600

# Alle initialverdiene er 0, utenom akselerasjonen
a[0] = 5

# Eulers Cromers metode:

for i in range(0,n-1):
    a[i] = 5 - c*(v[i])**2
    v[i+1] = v[i] + dt*a[i]
    x[i+1] = x[i] + dt*v[i+1]
    t[i+1] = t[i] + dt
    if x[i+1] > 100:
        print(np.argmax(t))
        break

# Plotte:
import matplotlib.pyplot as plt

ax1 = plt.subplot(311)
plt.title('Bevegelsesmodell for 100 m løp')
plt.plot(t[0:679],x[0:679])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Distanse[m]')

ax2 = plt.subplot(312, sharex = ax1)
plt.plot(t[0:678],v[0:678])
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Fart[m/s]')

ax3 = plt.subplot(313, sharex = ax1)
plt.plot(t[1:678],a[1:678])
plt.ylabel('Akselerasjon[$m/s^{2}$]')
plt.xlabel('t[s]')
plt.show()
