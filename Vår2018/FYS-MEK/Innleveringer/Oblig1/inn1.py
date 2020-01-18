### Første oblig FYS-MEK

# Oppgave e)
# Skal bruke eulers metode
# Initiering

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

# Oppgave f
# 100 meter på t sekunder
t1 = np.argwhere(x > 100)
x100 = t[t1]
print('tida ved x=100 er {:.3e}'.format(x100[0][0]))


# Dvs en snitthastighet på
v_ = 100/6.79
print(v_) #

### g) Numerisk verdi for terminalhastigheten

"""
v_t = np.amax(v)
v_ti = (np.argwhere(v < v_t))
print(v_ti)
t_vmaks = t[v_ti[0][-1]]
x_vmaks = x[int(t_vmaks)]
print('Terminalhastigheten er {:.2e} etter {:.2f} sek og {:} meter'.format(v_t,t_vmaks, x_vmaks))
# Terminalhastigheten er 2.58e+01 etter 6.79 sek

## g) Kan finne terminalhastighet når a = 0

a_min = np.amin(a)
print(a_min)
a_terminal = np.argwhere(a > a_min)
term_index = a_terminal[-1][0]
print(term_index)
tid = t[a_terminal[-1][0]]
print(tid)
x_term = x[term_index]
print(x_term)

# Umulig å finne ut fra modellen!!!!
"""
# h )
# OK

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

# j Hvor raskt han løper 100 m
t_100 = t1[i_max]
print('Han løper 100 m på {:.2f} sek'.format(t_100))
# Han løper 100 m på 9.31 sek


# i)
w = 1
def a_var2(t, v):
    return (F + fc*np.exp(-(t/tc)**2) \
            - fv*v - 0.5*A*(1-0.25*np.exp(-(t/tc)**2))\
            *rho*cd*(v-w)**2)/m


n = 1000            # tidssteg
T = 10              # sek
dt = T/n            # ett steg
a2 = np.zeros(n)
v2 = np.zeros(n)
x2 = np.zeros(n)
t2 = np.zeros(n)

for i in range(0,n-1):
    a2[i] = a_var2(float(t2[i]),float(v2[i]))
    v2[i+1] = v2[i] + dt*a2[i]
    x2[i+1] = x2[i] + dt*v2[i+1]
    t2[i+1] = t2[i] + dt
    if x2[i+1] > 100:
        break
i_max = np.argmax(t2)

print('Tida han bruker med 1 m/s vindhastighet medvind er {:.2f}'.format(t2[i_max]))
#Tida han bruker med 1 m/s vindhastighet er 9.21
#Tida han bruker med 1 m/s vindhastighet motvind er 9.43


# K
def Fc(t):
    return  fc*np.exp(-(t/tc)**2)
def Fv(v):
    return fv*v
def D(t,v):
    return 0.5*A*(1-0.25*np.exp(-(t/tc)**2))\
        *rho*cd*(v**2)

# Beregning av kreftene
F = [F]*len(t1)
Fct = Fc(t1[0:i_max])
Fvt = Fv(v1[0:i_max])
Dt = D(t1[0:i_max],v1[0:i_max])
t_intervall = t1[0:i_max]

# plot
plt.plot(t_intervall, F[0:i_max], label = 'Drivkraft')
plt.plot(t_intervall, Fct, label = 'Initiell drivkraft')
plt.plot(t_intervall,Fvt, label = 'Fysiologisk begrensning')
plt.plot(t_intervall,Dt, label = 'Luftmotstand')
plt.xlabel('Tid [s]')
plt.ylabel('Kraft [$N= kg*m*s^{-2}$]')
plt.legend(loc = 'center right')
plt.ylabel('Kraft [$N= kg*m*s^{-2}$]')
plt.show()
