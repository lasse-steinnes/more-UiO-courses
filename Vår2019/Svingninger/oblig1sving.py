## Oblig1
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,2*np.pi,100)

def func(t):
    p = 2*(-2.2*np.sin(2*t + 1.2))
    x = 1.1*np.cos(2*t + 1.2)
    return p, x

p, x = func(t)

plt.plot(x,p)
plt.axis("equal")
plt.show()
