import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,np.pi,1000)
plt.plot(t,np.cosh(t))
plt.xlabel('t/$\omega$')
plt.ylabel('cosh(t)/$r_{0}$')
plt.show()
