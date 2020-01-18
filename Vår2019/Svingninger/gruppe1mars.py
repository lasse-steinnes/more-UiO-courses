# Gruppeoppgaver
import numpy as np
import matplotlib.pyplot as plt

#t = np.linspace()
x = np.linspace(0,2*np.pi,1000)

def wave(x,t):
    A = 2
    k = 10
    omega = 4
    return A*np.cos(k*x - omega*t)

plt.plot(x,wave(x,0))
#oppgave c)
plt.plot(x,wave(x,2),'--', label = '2 sek')
plt.plot(x,wave(x,2.4),'--', label = '2.4')
plt.legend()
plt.show()

# oppgave b
k = 10
print(2*np.pi/k)

# oppgave c)
dt = 2.4 - 2
dx = 0.65 - 0.48
print('speed', dx/dt);

# Oppgave d
lambda_ = 2*np.pi/k
omega = 4
v_an =lambda_*omega/(2*np.pi)
print('Analytisk løsning',v_an)
