###### Oblig 1 MEK ######
import numpy as np
import matplotlib.pyplot as plt
# Oppgave 1
# Plotte skalerte funksjoner
t = np.linspace(0,1,1000)
theta1 = np.linspace(np.pi/6,np.pi/6,1000)
theta3 = np.linspace(np.pi/3,np.pi/3,1000)
theta2 = np.linspace(np.pi/4,np.pi/4,1000)

x = t
y = lambda theta,t: np.tan(theta)*(t-t**2)
plt.plot(x,y(theta1,t),x,y(theta2,t),x,y(theta3,t), '-m')
plt.xlabel(r'$x^{*}$')
plt.ylabel(r'$y^{*}$')
plt.legend([r'$\theta$ = $\frac{pi}{6}$',r'$\theta$ = $\frac{pi}{4}$',r'$\theta$ = $\frac{pi}{3}$'])
plt.title('Skalert kast')
plt.show()

# 2b)
# skal tegne konturlinja vha python
# v tangent til strømlinje, bestemmer retning


def streamline(n):
    '''Regner ut et grid og en strømfunksjon'''
    x= np.linspace(-5,5,n)
    [X,Y] = np.meshgrid(x,x)
    psi= Y - np.log(abs(x))
    return X, Y, psi

x0,y0,psi = streamline(100)

def vecfelt(n):
    x = np.linspace(-5,5,n)
    [X,Y] = np.meshgrid(x,x)
    Vx = X*Y
    Vy = Y
    return X,Y,Vx, Vy

x,y, vx, vy = vecfelt(25) # færre punkter for strømvektorene

# antall punkter - > antall piler
# scale bestemmer lengden av pilene


plt.contour(x0,y0,psi)
plt.colorbar()

# plotter vektorplot av hastighetsfelt
plt.quiver(x,y,vx,vy,scale = 20, units = 'x', color = 'r') #
plt.axis([-5,5,-5,5])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Oppgave 3e
# Skal plotte tilnærming for psi
def streamTaylor(n):
    x = np.linspace(-.5,.5,n)
    [X,Y] = np.meshgrid(x,x)
    return X, Y

x,y = streamTaylor(15)
x2,y2 = streamTaylor(30)

def psi(x,y):
    return 1 - 1/2*(x**2+y**2)

psiarr = psi(x,y) # tilnærmede konturlinjer rundt origo
psiarr2 = psi(x2,y2)
vy, vx = np.gradient(psi(x,y))
plt.quiver(x,y,-vx,vy,units = 'x',color = 'r') # bruker def a strømfunksjonen

plt.contour(x2,y2,psiarr2)
plt.colorbar()
plt.axis([-0.5,0.5,-0.5,0.5])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Oppgave 4
# a) Skal plotte konturlinjer for psi
# Hvert sitt diagram for n = 5, n = 30

def streamfun(n):
    x= np.linspace(-0.5*np.pi,0.5*np.pi,n)
    #resultatet er en vektor med n elementer, fra -pi/2 til pi/2
    [X,Y] = np.meshgrid(x,x)
    psi= np.cos(X)*np.cos(Y)
    return X, Y, psi

# for n = 5;
x,y,psi = streamfun(5)

# for n = 30
x1,y1,psi1 = streamfun(30)

# plotte konturlinjer til skalarfeltet (strømlinje)
import matplotlib.pyplot as plt
plt.contour(x,y,psi)
plt.colorbar()
plt.axis('equal')
plt.title('n = 5')
plt.show()

plt.contour(x1,y1,psi1)
plt.colorbar()
plt.axis('equal')
plt.title('n = 30')
plt.show()


# b) Skal plotte vektorfeltene

def velfield(n):
    x = np.linspace(-0.5*np.pi,0.5*np.pi,n)
    [X,Y] = np.meshgrid(x,x)
    Vx = np.cos(X)*np.sin(Y)
    Vy = -np.sin(X)*np.cos(Y)
    return X,Y,Vx, Vy


x,y,vx, vy = velfield(15)

# plotter vektorplot av hastighetsfelt
plt.quiver(x,y,vx,vy,scale = 5, units = 'x', color = 'r')
plt.axis([-0.5*np.pi,0.5*np.pi,-0.5*np.pi,0.5*np.pi])
plt.xlabel(r'x($\theta$)')
plt.ylabel(r'y$(\theta$)')
plt.show()

# vektorplot og kontur
plt.quiver(x,y,vx,vy,scale = 5, units = 'x', color = 'r')
plt.contour(x1,y1,psi1)
plt.axis([-0.5*np.pi,0.5*np.pi,-0.5*np.pi,0.5*np.pi])
plt.xlabel(r'x($\theta$)')
plt.ylabel(r'y$(\theta$)')
plt.show()
