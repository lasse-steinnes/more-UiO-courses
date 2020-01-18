### Oppgave 1B8
### 'lander'
### jobbe i km/t istedet for
# Henter ut initiell posisjon og hastighet
import numpy as np
import matplotlib.pyplot as plt

from AST2000SolarSystemViewer import AST2000SolarSystemViewer
seed = 61252; system = AST2000SolarSystemViewer(seed)

x0 = system.x0          # initiell pos x
y0 = system.y0          # initiell pos y
vx0 = system.vx0        # initiell vel x
vy0 = system.vy0        # initiell vel y
radius = system.radius
stm = system.star_mass # masse til stjerne
M = system.mass;           # [solare masser] massene til planetene
G = 4 * np.pi * np.pi   # Gravitasjonskonstant i astronomiske enheter
# Benytt år som tid, AU som distanse

### velger å lande på planet 2:
m1 = M[1]*2*(10**30)          # masse planet 2 [kg]
m2 = 100                      # masse satelitt
rho0 = system.rho0[1]         # kg/m^3
radius = radius[1]*1000      # radius planet km --> m
G2 =  6.674*10**(-11) #N·kg–2·m2  SI units

def rho(h):
    g = G2*m1/(radius**2)
    h_scale = 75200/(g*m2)
    return rho0*np.exp(-h/h_scale)
print(m1,m2,rho0)
###
L = np.linspace(0,100)
A = L**2  # mellom 1 og 500
print(A[3])
pos_over = 40000*1000 + radius #[m] #

pos = [0,pos_over]
vel = [0.005,0.0]                      #ønsker å ha array s.a x, y komp i kolonne
#print(vel[1])

T = 10**6 # burde ta et par dager

from numba import jit
@jit

def lander(T,aval, rho_f, pos0,vel0, radius):
    dt = 0.1
    steps = int(T/dt)
    t = np.zeros(steps)

    pos = np.zeros(shape = (steps,2)); pos[0,:] = pos0
    vel = np.zeros(shape = (steps,2)); vel[0,:] = vel0
    #plt.ion()
#    for i in range(steps-1):

# arundo
    v_r = 0
    i = 0
    while np.sqrt(pos[i,0]**2+pos[i,1]**2) > radius:
        r_abs = np.sqrt(pos[i,0]**2+pos[i,1]**2)
        vel_abs = np.sqrt(vel[i,0]**2 + vel[i,1]**2)
        h = r_abs - radius
        print('posx {:}, posy {:}'.format(pos[i,0],pos[i,1]))
        F = -G2*m1*m2/(r_abs**3)*pos[i,:]
        F_r = -G2*m1*m2/(r_abs**2)
        F_Dr = 0.5*aval*rho_f(h)*v_r**2
        F_D = -0.5*aval*rho_f(h)*vel_abs**2*vel[i,:]/np.sqrt(vel[i,0]**2 + vel[i,1]**2)
        a = (F + F_D)/m2
        a_r = (F_r + F_Dr)/m2
        v_r = v_r + dt*a_r
        vel[i+1,:] = vel[i,:] + dt*a
        pos[i+1,:] = pos[i,:] + dt*vel[i+1,:]
        t = t + dt
        print(i)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        #plt.plot(pos[i,0],pos[i,1],'o')
        #plt.pause(0.00001)
        if v_r > 3:
            print('Satelitt har krasjet pga. høy hastighet v:{:}, vx: {:}, vy:{:}'.format(np.sqrt(vel[i,0]**2+vel[i,1]**2),vel[i,0],vel[i,1]))
            break
        elif np.linalg.norm(F_D) > 25000: # N
            print('Satelittens fallskjerm ble ødelagt pga. høy luftmostand')
            break
        elif radius + 10**(-1) > np.sqrt(pos[i,0]**2+pos[i,1]**2):
            print('Satelitt har landet med A = {:}'.format(aval))
            break
        i = i + 1

lander(T,A[30],rho,pos,vel,radius)

"""
Kjøreeksempel
----------------------------
37.4843815077  (areal fallskjerm)
posx 0.0, posy 42054392.440677114
0
posx 0.0005, posy 42054392.44059449
1
posx 0.0009999999999990177, posy 42054392.44042923
2
posx 0.0014999999999960704, posy 42054392.440181345
3
[...]
"""
