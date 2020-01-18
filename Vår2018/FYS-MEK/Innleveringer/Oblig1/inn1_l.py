# l)
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
