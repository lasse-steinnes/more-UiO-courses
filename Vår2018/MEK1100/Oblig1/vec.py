# Oppgave 4b
# Skal bruke velfield-program til å plotte hastighetsvektorer

from velfield import velfield
import numpy as np
import matplotlib.pyplot as plt

# Lager arrayene
x,y,vx, vy = velfield(15)

# plotter vektorplot av hastighetsfelt
plt.quiver(x,y,vx,vy,scale = 5, units = 'x', color = 'r')
plt.axis([-0.5*np.pi,0.5*np.pi,-0.5*np.pi,0.5*np.pi])
plt.xlabel(r'x($\theta$)')
plt.ylabel(r'y($\theta$)')
plt.show()
