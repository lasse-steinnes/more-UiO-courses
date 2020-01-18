import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
"""
nx, ny = (3, 3)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
print(xv)
print(yv)
"""

startx = -2500 #Define where the parts of the clouds should start/stop in the x direction
endx = 2500

startz = -2500 #Define where the cloud should start/stop in the z direction
endz = 2500

negheight = 6000 #set the negative cloud at a height (in the y direction) of 6000m
posheight = 80000 #set the positive cloud at a height of 8000m
belowground = 750 # eventuelt kan jeg endre på dette direkte i utregning

Q = -15 #Q is the total charge on the bottom (negative) part of the cloud
Q2 = 15 #Q2 is the total charge of the top (positive) part of the cloud
Q3 = 1*10**(-1)  #Q3 is total charge on the ground due to polarization

k = 9e9 #Coulomb's constant


pos = np.array([0,0,0]) #the observation position (start at 0,0,0)

# ønsker å konstruere array med flere slike verdier, slik at E-feltet kan beregnes utover et grid,
#slik kan man finne verdier der lynet kan slå ned
# Poeng: Du må ha polarisering i bakken, som bidrar til E-feltet.
n = 100 # steps
pos1 = np.zeros(shape =(n,3))
x_pos = np.linspace(-2500,2500,n)
z_pos = np.linspace(-2500,2500,n)
y1 = np.zeros((n,n))

x, z = np.meshgrid(x_pos,z_pos, indexing = 'ij')
#print(x[0]) # varierer x mellom -2500,og 2500
#print(z[0]) # holder z = -2500 konstant

y_pos = (x**2)/(10**4.5) - (z**2)/(10**4.5) + 1200 # som en sadel
#print(y_pos[0,:]) # får verdier etter x pos i og y pos j
# da har jeg overflaten: kan jeg plotte den?
#plt.contourf(x,z,y_pos, cmap = 'jet')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, z, y_pos)
plt.show()

# Bra, så for flate
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, z, y1)
plt.show()

# for flaten
fig = plt.figure()
n = 100
x_pos = np.linspace(-150,150,n)
z_pos = np.linspace(-150,150,n)
y1 = np.zeros((n,n))

x, z = np.meshgrid(x_pos,z_pos, indexing = 'ij')


ax = fig.gca(projection='3d')
ax.plot_surface(x, z, y1,alpha = 1.0,color = 'grey')
    # for lynet
cyl_a = 10
x= np.linspace(0, 10, 100)
z= np.linspace(0, negheight, 100)
Xcyl, Zcyl=np.meshgrid(x, z)
Yc = np.sqrt(cyl_a**2-Xcyl**2)
rstride = 20
cstride = 10
ax.plot_surface(Xcyl, Yc, Zcyl,alpha=0.7, color = 'yellow', rstride=rstride, cstride=cstride)
ax.plot_surface(Xcyl, -Yc, Zcyl, alpha = 0.7, color ='yellow', rstride=rstride, cstride=cstride)

    # for dødelig radius
groundheight = 0
cyl_a = 74
x= np.linspace(0, cyl_a, 100)
z= np.full((n,n),groundheight+1)
Xcyl, Zcyl=np.meshgrid(x, z)
Yc = np.sqrt(cyl_a**2-Xcyl**2)
rstride = 20
cstride = 10
ax.plot_surface(Xcyl, Yc, Zcyl,alpha=0.7, color = 'red', rstride=rstride, cstride=cstride)
ax.plot_surface(Xcyl, -Yc, Zcyl, alpha = 0.7, color ='red', rstride=rstride, cstride=cstride)

plt.show()


nx = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)
nz = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)

stepx = (endx - startx)/nx #Define the spacing between each chunk in the x direction
stepz = (endz - startz)/nz #Define the spacing between each chunk in the z direction

dQ = Q/(nx*nz) #Charge of each chunk of the the negative part of the cloud: Denne vil få negativ verdi
dQ2 = Q2/(nx*nz) #Charge of each chunk of the positive part of the cloud
dQ3 = Q3/(nx*nz)
print('Ladningstetthet for overflaten er {:.2e} (øvre skylag), {:} (nedre skylag) og {:} (substrat)'.format(dQ,dQ2,dQ3))

# For valley
# ønsker å vektorisere finne ut hvor det mest sannsynlig slår ned

"""
def efield():
    @jit(nopython=True)
    def efieldsum(m,p):
        efield = 0
        pos = np.array([x[m,p],y_pos[m,p],z[m,p]]) # høyde h
        for i in range(0,nx): #iterate over the x dimension of the cloud
            xloc = startx + i*stepx
            for j in range(0,nz): #iterate over the z dimension of the cloud
                zloc = startz + j*stepz

                posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
                negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
                groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
                efield = efield + negfield + posfield + groundfield
        return efield

    for m in range(n):
        for p in range(n):
            E_arr[m,p] = efieldsum(m,p)
    return E_arr

E_arr =  efield(0)
print(E_arr)
"""

### Duplicate of code, just with variable height, but straight plane
# ønsker å vektorisere finne ut hvor det mest sannsynlig slår ned.
"""
E_arr = np.zeros((n,n))
efield = 0


def efield(h):
    @jit(nopython=True)
    def efieldsum(m,p):
        efield = 0
        pos = np.array([x[m,p],h,z[m,p]]) # høyde h
        for i in range(0,nx): #iterate over the x dimension of the cloud
            xloc = startx + i*stepx
            for j in range(0,nz): #iterate over the z dimension of the cloud
                zloc = startz + j*stepz

                posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
                negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
                groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
                efield = efield + negfield + posfield + groundfield
        return efield

    for m in range(n):
        for p in range(n):
            E_arr[m,p] = efieldsum(m,p)
    return E_arr

E_arr =  efield(0)
print(E_arr)

#print(E_arr)
#        if efield < -3*10**6:
#            print('Lightning!! E-field value:{:}, position: {:}'.format(efield,pos))
#print(E_arr)
### Duplicate of code, just with point
"""

efield = 0


pos = np.array([0,50,0])
for i in range(0,nx): #iterate over the x dimension of the cloud
    xloc = startx + i*stepx
    for j in range(0,nz): #iterate over the z dimension of the cloud
        zloc = startz + j*stepz

        negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
        posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
        groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
        efield = efield + negfield + posfield + groundfield

if efield < -3*10**6:
    print('Lightning!! E-field value:{:}'.format(efield))

print("The e-field at the observation position is", efield, "Newtons per coulomb")


"""
# strømmen som vi gå gjennom kroppen
def find_stream(V,r,rmax,s20,s100):
    I_new = np.zeros(steps)
    for i in range(steps):
        R = 171e-2 #Ω/m # R = rho*x/A # lever
        I_new[i] = V[i]/R
    return I_new
"""
