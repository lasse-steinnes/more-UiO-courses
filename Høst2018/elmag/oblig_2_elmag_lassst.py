### Obligatorisk innlevering 2, oppgave 1 ###
# Skal skrive et program som beregner bevegelsen til staven

import numpy as np
import matplotlib.pyplot as plt
"""
n = 10000            # tidssteg
T = 14              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a_arr = np.zeros(n)     # Array for akselerasjon
v = np.zeros(n)     # Array for fart
x = np.zeros(n)     # Array for x posisjon
t = np.zeros(n)     # Tidsarray

# Initialverdiene
# alle er 0, bortsett fra v0
v[0] = 20 # m/s
# B mellom 10 og 10**3
B = 10**(-1.03)
rho = 4.6*10**(-7) # tilsvarende elektisk stål
r = 0.15 # radius til stålsylinder
# lag array med mulige b-verdier. Skal stoppe innen 100 m.
A = np.pi*r**2
l = 2 # meter bredde på vogn
m = 2000 #vogn veier 2 tonn
p_l = 72
def Res(x):
    return rho*x/A

def a(Res_funk,v,B):        # p_l er plass til ledningen
    return -(B**2)*(l**2)*v/(Res_funk*m)
# Eulers Cromers metode:
for i in range(0,n-1):
    if x[i] < p_l:
        a_arr[i] = 0
    else:
        a_arr[i] = a(Res(x[i]-p_l),v[i],B)
    v[i+1] = v[i] + dt*a_arr[i]
    x[i+1] = x[i] + dt*v[i+1]
    t[i+1] = t[i] + dt

if x[-1] >= 100:
    print('Vogn har krasjet med B:{:}'.format(B))

elif -1*10**(-1) <= v[-1] <=  1*10**(-1):
    print('Vogn har blitt nedbremset til stopp, med B-felt: {:.4e}'.format(B))

plt.figure()
plt.subplot(121)
plt.plot(t,v)
plt.xlabel('tid [s]')
plt.ylabel('Hastighet [m/s]')

plt.subplot(122)
plt.plot(t,x)
plt.xlabel('tid [s]')
plt.ylabel('Posisjon [m]')
plt.show()

#### Oblig 2 del 2 ####
# simulere strøm i en nervecelle
# oppgave c
# For å løse oppgaven numerinsk velges

### initialbetingelser og konstanter
# g(t=0) = 0
r1 = 100
r2 = 250 # 250 mega ohm/sek
v_s = 70*10**(-2) #* -6
c = 0.01 #0.01*np.pi*(10**(-5)/2)**2


def qI_num(T,n):
    q = np.zeros(n)
    dq = np.zeros(n)
    I2 = np.zeros(n)
    t = np.zeros(n)
    dt = T/n

    for i in range(n-1):
        dq[i] = 1/r2*(v_s -(1/c +r2/(c*r1))*q[i])
        q[i+1] = q[i] + dq[i]*dt
        I2[i] = dq[i] + q[i]/(c*r1)
        t[i+1] = t[i] + dt
    I2[-1] = I2[n-2]
    return t,q,I2

t,q, I2 = qI_num(10,n)


t1 = np.linspace(0,10,n)

def qI_an(t):
    q = c*r1*v_s/(r1+r2)*(1-np.exp(-t*(r1+2)/(c*r1*r2)))
    dq = v_s/r2*np.exp(-t*(r1+r2)/(c*r1*r2))
    I2 = q/(c*r1) + dq
    return q, I2

plt.figure()
qnum, I2num = qI_an(t1)
plt.plot(t1,qnum, label = 'Numerisk')
plt.plot(t,q, label = 'Analytisk')
plt.legend()
plt.xlabel('tid [s]')
plt.ylabel('Ladning [C]')
plt.show()

plt.figure()
plt.plot(t1,I2num,label = 'Numerisk')
plt.plot(t,I2, label = 'Analytisk')
plt.legend()
plt.xlabel('tid [s]')
plt.ylabel('strøm [A]')
plt.show()
"""
## oppgave d og videre
##
R1 = 1*10**8
R2 = 10**6
C = 10**(-12)
v_s = 100*10**(-3)
n = 100
m = 10**4
T = 0.00005 # s
l = 100*10**(-3)
light = 3*10**8

# g) Løsning av ligningene gir en simulering av spenningen for hver kondensator

def vc_num(vs, T,n,m):
    V = np.zeros(shape = (n,m))
    V[0,0] = vs
    t = np.zeros(m)
    dt = T/m

    for k in range(n-1):
        for j in range(m-1):
            V[k,j+1] = V[k,j] + (1/C)*((V[k+1,j] - 2*V[k,j] + \
            V[k-1,j])/R2 - V[k,j]/R1)*dt;
            t[j+1] = t[j] + dt
    for i in range(n):
        plt.plot(t,V[i,:])
        plt.xlabel('t [s]')
        plt.ylabel('V')

    plt.show()
    print('V_inn = {:.4e}, V_ut = {:.4e}'.format(vs,np.amax(V[98,:])))
    k = np.where(V[98,:] == np.amax(V[98,:])) # np.amax(V[98,:]))
    print(k)
    print('Signaltida er: {:.4e} s, dvs. en hastighet på {:.4e} m/s, som er {:.2e}x lyshastigheten'.format(t[k[0][0]],l/t[k[0][0]],(l/t[k[0][0]])/light))
    return V

V = vc_num(v_s,T,n,m)

def vsfunk(t):
    global v_s
    if T/100 < t < T/50:
        vsign = v_s
    elif T/40 < t < T/30:
        vsign = v_s
    elif T/25 < t < T/20:
        vsign = v_s
    elif T/15 < t < T/12:
        vsign = v_s
    else:
        vsign = 0
    return vsign

# h) Modifisering av programmet slik at Vt start og vt slutt leses av
T = 0.00005
def vcmod_num(vs,vsfunk, T,n,m):
    V = np.zeros(shape = (n,m),dtype = np.float64)
    Vsignal = np.zeros(shape = (n,m),dtype = np.float64)
    t = np.zeros(m,dtype = np.float64)
    dt = T/m
    for k in range(n-1):
        for j in range(m-1):
            Vsignal[0,j] = vsfunk(t[j])
            Vsignal[k,j+1] = Vsignal[k,j] + (1/C)*((Vsignal[k+1,j] - 2*Vsignal[k,j] + \
            Vsignal[k-1,j])/R2 - Vsignal[k,j]/R1)*dt
            t[j+1] = t[j] + dt
    for i in range(n):
        plt.plot(t,Vsignal[i,:])
        plt.xlabel('tid [s]')
        plt.ylabel('V')
        plt.axis([0,0.00001,0,0.12])
    plt.show()

    plt.plot(t,Vsignal[0,:])
    plt.plot(t,Vsignal[98,:])
    plt.xlabel('tid [s]')
    plt.ylabel('V')
    plt.axis([0,0.00001,0,0.12])
    plt.show()

    print('V_inn = {:.4e}, V_ut = {:.4e}'.format(vs,np.amax(Vsignal[98,:])))
    return t,Vsignal

#def v_s(t):

t,V = vcmod_num(v_s,vsfunk,T,n,m)

# i) Signalhastighet
k = np.where(V[98,:] == np.amax(V[98,:])) # np.amax(V[98,:]))
print(k)
print('Signaltida er: {:.4e} s, dvs. en hastighet på {:.4e} m/s, som er {:.2e}x lyshastigheten'.format(t[k[0][0]],l/t[k[0][0]],(l/t[k[0][0]])/light))
#####################
"""
Vogn har blitt nedbremset til stopp, med B-felt: 9.3325e-02
V_inn = 1.0000e-01, V_ut = 5.3343e-28
Signaltida er: 4.9000e-07 s, dvs. en hastighet på 2.0408e+05 m/s, som er 6.80e-04x lyshastigheten

"""
