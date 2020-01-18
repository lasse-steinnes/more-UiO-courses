## Besvarelse til AST 1B7 og 1B8 ##
# Henter ut initiell posisjon og hastighet
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

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
#print(M)
#print(stm)
#print(x0) # åtte planeter

T = 1000
dt = 10**(-1)
def Euler_cromer(T,dt,x0,y0,vx0,vy0):  # må beregne god tid
    n = 8
    steps = int(T/dt)
    t = np.zeros(steps)
    posx = np.zeros(shape = (steps,n)); posx[0,:] = x0;
    posy = np.zeros(shape = (steps,n)); posy[0,:] = y0;
    velx = np.zeros(shape = (steps,n)); velx[0,:] = vx0;
    vely = np.zeros(shape = (steps,n)); vely[0,:] = vy0;
    ax = np.zeros(shape = (1,n)); ay = np.zeros(shape = (1,n))
    for i in range(0,steps-1):
         ay = -G*stm*posy[i,:]/(posx[i,:]**2 + posy[i,:]**2)**(3/2)
         ax = -G*stm*posx[i,:]/(posx[i,:]**2 + posy[i,:]**2)**(3/2)
         velx[i+1,:] = velx[i,:] + dt*ax
         vely[i+1,:] = vely[i,:] + dt*ay
         posx[i+1,:] = posx[i,:] + dt*velx[i+1,:]
         posy[i+1,:] = posy[i,:] +dt*vely[i+1,:]
         t[i+1] = t[i] + dt
#         if posx[0,7] - 10**(-8) <= posx[i,7] <= posx[0,7] +10**(-8) and\
#            posy[0,7] - 10**(-8) <= posy[i,7] <= posy[0,7] +10**(-8):
#            break
    return posx,posy,velx,vely,t

posx,posy,velx,vely,t = Euler_cromer(T,dt,x0,y0,vx0,vy0)
#print(posx)
#print(posx[0].value[0])

steps = int(T/dt)
planet_pos = np.zeros(shape = (2,8,steps))
planet_pos[0,:,:] = np.transpose(posx)
planet_pos[1,:,:] = np.transpose(posy)
times = t

system.orbit_xml(planet_pos,times) #planet_pos = (2,n_planets, n_times)


# kan plotte mot analytisk og se hvordan disse passer
for i in range(8):
    plt.plot(planet_pos[0,i,:],planet_pos[1,i,:],label ='m:{:}'.format(M[i]))

#plt.legend()
plt.ylabel('AU')
plt.xlabel('AU')
plt.show() # jippi


# Kan ev. plotte velocities mhp posisjonen for hver av plantene
#for i in range(8):
#plt.plot(posx[1,:],velx[1,:])
#plt.show()
