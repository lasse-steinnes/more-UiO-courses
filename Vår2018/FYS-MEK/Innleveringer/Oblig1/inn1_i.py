## i) Modifisere modellen til å inkludere:
# I) endring i overflate
# II) fysiologisk begrensning
# III) Endring i krefter som kan utgjøres på bakken pga. endring i kroppsstilling


F = 400
fc = 488
tc = 0.67
fv = 25.8
A = 0.45
rho = 1.293
cd = 1.2
m = 80
def a_var(t, v):
    return (F + fc*np.exp(-(t/tc)**2) \
            - fv*v - 0.5*A*(1-0.25*np.exp(-(t/tc)**2))\
            *rho*cd*(v**2))/m


n = 1000            # tidssteg
T = 10              # sek
dt = T/n            # ett steg
a1 = np.zeros(n)
v1 = np.zeros(n)
x1 = np.zeros(n)
t1 = np.zeros(n)

for i in range(0,n-1):
    a1[i] = a_var(float(t1[i]),float(v1[i]))
    v1[i+1] = v1[i] + dt*a1[i]
    x1[i+1] = x1[i] + dt*v1[i+1]
    t1[i+1] = t1[i] + dt
    if x1[i+1] > 100:
        break
i_max = np.argmax(t1)

import matplotlib.pyplot as plt
ax1 = plt.subplot(311)
plt.title('Modifisert bevegelsesmodell for 100 m løp')
plt.plot(t1[0:i_max],x1[0:i_max])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Distanse[m]')

ax2 = plt.subplot(312, sharex = ax1)
plt.plot(t1[0:i_max],v1[0:i_max])
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Fart[m/s]')

ax3 = plt.subplot(313, sharex = ax1)
plt.plot(t1[0:i_max],a1[0:i_max])
plt.ylabel('Akselerasjon[$m/s^{2}$]')
plt.xlabel('t[s]')
plt.tight_layout()
plt.show()
