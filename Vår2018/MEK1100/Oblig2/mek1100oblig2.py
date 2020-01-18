### innlevering 2 ###

# Oppgave a)
# Matrisene X,Y
# u,v
# xit, yit

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Dette virker med versjon 7 MAT-filer
data = sio.loadmat("data.mat")
x = data.get("x")
y = data.get("y")
u = data.get("u")
v = data.get("v")
xit = data.get("xit")
yit = data.get("yit")

xlen = np.shape(x); ylen = np.shape(y);ulen = np.shape(u);
vlen = np.shape(v); xitlen = len(xit); yitlen = len(yit)
datapunkter = (xlen,ylen,ulen,vlen)
print('----------------------')
print('Lengden til array')
print('----------------------')
for i in datapunkter:
    print('#datapunkter: {:}'.format(i))
print('----------------------')

# skal sjekke antall datapunkter for vektorene
print('# datapunkter for vektorene x:{:}, datapunkter y:{:}'.format(np.shape(xit), np.shape(yit)))
# en akse (rad), 194 punkter.


print(x) # y-ene holdt konstant (y:rader, x:kolonner)
print(y) # x-ene holdt konstant   # Sammen utgjør disse et grid

def Test_matrise(x):
    n = 201
    m = 194
    for j in range (1,n):
        for i in range(1,m):
            if x[j,i] - x[j-1,i-1] == 0.5: # eventuelt if not
                continue
            else:
                print("Ujevn skala")
                break

Test_matrise(x)
Test_matrise(y)     # Ok Kjøre "stille"

# Sjekk at lengde til y-koord er 100 mm : summere element i matrise y
# kan sjekke ved å summere de to siste leddene
diameter = abs(y[0,0]) + abs(y[-1,-1])
print('Diameteren er {:.2f}. mm'.format(diameter))

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
----------------------
Lengden til array
----------------------
#datapunkter: (201, 194)
#datapunkter: (201, 194)
#datapunkter: (201, 194)
#datapunkter: (201, 194)
----------------------
# datapunkter for vektorene x:(1, 194), datapunkter y:(1, 194)
[[  0.    0.5   1.  ...,  95.5  96.   96.5]
 ...,
 [  0.    0.5   1.  ...,  95.5  96.   96.5]]

[[-50.  -50.  -50.  ..., -50.  -50.  -50. ]
 ...,
 [ 50.   50.   50.  ...,  50.   50.   50. ]]

Diameteren er 100.00. mm
"""

# Oppgave b
# skal lage konturlinjer (med vektorpiler)
# Skal jobbe med u og v, lage for hver?
#plt.contour()
#plt.colorbar()

# plotter vektorplot av hastighetsfelt
v_ = np.sqrt(u**2 + v**2)
plt.figure(figsize = (12,5))

lvl1 = np.linspace(0,500,10)
ax1 = plt.subplot(121)
plt.contourf(x,y,v_, levels = lvl1, cmap = 'jet')
plt.plot(xit,yit,'*r')
plt.colorbar()
plt.ylabel('y [mm]')
plt.xlabel('x [mm]')

lvl2 = np.linspace(500,np.max(v_))
ax2 = plt.subplot(122)
plt.xlabel('x [mm]')
plt.contourf(x,y,v_, levels = lvl2, cmap = 'jet')
plt.colorbar()
plt.plot(xit,yit,'*r')


plt.suptitle(r'Konturlinjer til farten: $\sqrt{u^{2} + v^{2}}$')  # levels konturer nivå : 0-500, 500-maxv
plt.show()

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
(plot)
"""
# Oppgave c)
plt.quiver(x[::10,::10],y[::10,::10],u[::10,::10],v[::10,::10],units = 'x', color = 'r')
plt.plot(xit,yit, '*r')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
# tre rektangler:
def plot_rektangel(xmin,xmax,ymin,ymax): # (x1,x2)(y1,y2)
    plt.plot((xmin,xmin),(ymin,ymax),'black')
    plt.plot((xmin,xmax),(ymin,ymin),'red')
    plt.plot((xmax,xmax),(ymin,ymax),'green')
    plt.plot((xmin,xmax),(ymax,ymax),'blue')

plot_rektangel(x[159,34],x[159,69],y[159,34],y[169,34])
plot_rektangel(x[84,34],x[84,69],y[84,34],y[99,34]) #xmin,xmax,ymin,ymax
plot_rektangel(x[49,34],x[49,69],y[49,34],y[59,34])
plt.show()

# start:stop:step

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
(plot)
"""

# d: regne ut divergensen
# Vi har at divergens er lik
n = 201
m = 194
div = np.zeros((n,m))
for j in range (1,n):
    for i in range(1,m):
        div[j,i] = (u[j,i] - u[j,i-1])/(x[j,i] - x[j-1,i-1])\
                + (v[j,i] - v[j-1,i])/(y[j,i] - y[j-1,i-1])
#print(div)

# konturlinjene til divergensen

plt.contourf(x,y,div, cmap = 'jet')
plt.plot(xit,yit,'*r')
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plot_rektangel(x[159,34],x[159,69],y[159,34],y[169,34])
plot_rektangel(x[84,34],x[84,69],y[84 ,34],y[99,34]) #xmin,xmax,ymin,ymax
plot_rektangel(x[49,34],x[49,69],y[49,34],y[59,34])
plt.show()

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
(plot)
"""
# Oppgave e) Virvlingen i rundt z-aksen
curl = np.zeros((n,m))

for j in range (1,n-1):
    for i in range(1,m-1):
        curl[j,i] =  (v[j,i+1] - v[j,i-1])\
                - (u[j+1,i] - u[j-1,i])
# velger midtpunkter

#print(curl)
plt.contourf(x,y,curl, cmap = 'jet')
plt.plot(xit,yit,'*r')
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('y[mm]')
plot_rektangel(x[159,34],x[159,69],y[159,34],y[169,34])
plot_rektangel(x[84,34],x[84,69],y[84 ,34],y[99,34]) #xmin,xmax,ymin,ymax
plot_rektangel(x[49,34],x[49,69],y[49,34],y[59,34])
plt.streamplot(x,y,u,v, color = "blue") # plt.streamline()
plt.show()

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
(plot)
"""
# Oppgave f) Stokes og greens sats på rektanglene
# 1) Sirkulasjon regnet som kurveintegral
print(np.shape(v)) #201,194 y, x

def sirkulasjon(jmin,jmax,imin,imax):
    int_list = [0,0,0,0]
    for i in range(imin,imax+1):
        int_list[0] += u[jmin,i]*0.5
    for j in range(jmin,jmax+1):
        int_list[1] += v[j,imax]*0.5
    for i in range(imin,imax+1):
        int_list[2] -= u[jmax,i]*0.5
    for j in range(jmin,jmax+1):
        int_list[3] -= v[j,imin]*0.5
    return int_list[0], int_list[1], int_list[2], int_list[3], sum(int_list)

shrek1 = sirkulasjon(159,169,34,69)
shrek2 = sirkulasjon(84,99,34,69)
shrek3 = sirkulasjon(49,59,34,59)
print(shrek1)
print(shrek2)
print(shrek3)

# 2 ) Sirkulasjonen som flateintegral
def sirksurf(jmin,jmax, imin,imax):
    sum_ = 0
    curlarea = curl*0.5*0.5
    sum_curlarea = 0

    for j in range (jmin,jmax+1):
        for i in range(imin,imax+1):
            sum_curlarea += curlarea[j,i]
    return(sum_curlarea)


sirksurf1 = sirksurf(159,169,34,69)
sirksurf2 = sirksurf(84,99,34,69)
sirksurf3= sirksurf(49,59,34,69)
print(sirksurf1,sirksurf2,sirksurf3)

""" Kjøreeksempel
Lasse$ python mek1100oblig2.py
(70100.523878614273, 266.27357615858688, -68332.856099786746, 661.57273770969914, 2695.5140926958193)
(198.47559740489203, 300.21661027011692, -61243.464778495952, -231.82759129461652, -60976.600162115559)
(3798.5759031398288, 163.60937776314384, -4009.8351655432498, 78.302877021285482, 30.652992381008559)

2621.55869628 -61482.5409893 -12.2143338642
"""

# g) gauss sats: integrert fluks
# direkte:
def fluks_area(jmin,jmax,imin,imax):
    int_list = [0,0,0,0]
    for i in range(imin,imax+1):
        int_list[0] -= v[jmin,i]*0.5 #(x[jmin,i+1] - x[jmin,i])
        int_list[1] += v[jmax,i]*0.5
    for j in range(jmin,jmax+1):
        int_list[2] += u[j,imax]*0.5
        int_list[3] -= u[j,imin]*0.5
    return int_list[0],int_list[1],int_list[2],int_list[3], sum(int_list)

print(fluks_area(159,169,34,69))
print(fluks_area(84,99,34,69))
print(fluks_area(49,59,34,69))

# Kjøreeksempel
"""
Lasse$ python mek1100oblig2.py

(1556.8679439413959, -2059.6771847938708, 21664.567474322168, -21056.905628561482, 104.8526049082102)
(-5187.5640330678907, -4074.0522144394345, 14782.532896182345, -11997.85583077298, -6476.9391820979599)
(-195.5701479258336, 284.9436464350764, 1536.8217966413547, -1750.7639611955597, -124.56866604496213)
"""
