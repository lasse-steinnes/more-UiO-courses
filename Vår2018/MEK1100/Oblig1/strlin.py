# Oppgave 4
# a) Skal plotte konturlinjer for psi
# Hvert sitt diagram for n = 5, n = 30
import numpy as np

def streamfun(n):
    '''Regner ut et grid og en strømfunksjon'''
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
plt.figure(figsize = (8,5))

plt.subplot(121)
plt.contour(x,y,psi)
plt.xlabel(r'x($\theta$)')
plt.ylabel(r'y($\theta$)')
plt.axis('equal')
plt.title('n = 5')


ax2 = plt.subplot(122)
plt.contour(x1,y1,psi1)
plt.colorbar()
plt.axis('equal')
plt.xlabel(r'x($\theta$)')
plt.setp(ax2.get_yticklabels(), visible=False)
plt.title('n = 30')
plt.show()
