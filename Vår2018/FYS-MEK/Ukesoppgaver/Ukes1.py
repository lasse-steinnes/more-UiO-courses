### Oppgave 1.c: Skal plotte
import numpy as np
import matplotlib.pyplot as plt
A = 3 # akselerasjonskonstant [m/s**3]
def a(t):
    return A*t
def v(t):
    return 0.5*A*t**2
def x(t):
    return 1/6*A*t**3

# t-array
t = np.linspace(0,20,10000)

# plot

plt.subplot(3,1,1)
plt.plot(t,a(t))

plt.subplot(3,1,2)
plt.plot(t,v(t))

plt.subplot(3,1,3)
plt.plot(t,x(t))
plt.show()
