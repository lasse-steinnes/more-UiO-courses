## oppgave 3b ##
# Skal plotte venstre og høre side for å sammenlikne
import numpy as np
import matplotlib.pyplot as plt


left =  lambda theta: 16*np.sin(theta)**5
right = lambda theta: np.sin(5*theta) - 5*np.sin(3*theta) + 10*np.sin(theta)

theta = np.linspace(0,2*np.pi,100)

plt.plot(theta,left(theta),label = '$16\sin^{5}{\phi}$')
plt.plot(theta,right(theta),'--',label = '$\sin{5\phi} - 5\sin{3\phi} + 10sin{\phi}$')
plt.legend()
plt.xlabel('$\phi$')
plt.ylabel('f($\phi$)')
plt.show()
