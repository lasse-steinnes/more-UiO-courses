## Bølgefunksjonen ##
import numpy as np
import matplotlib.pyplot as plt

# konstanter
ha = 6.58*10**(-16) # [eV s], plancks reduserte konstant
a = 1 # nm, lengde på boksen
m = 0.511 # eV/c^2 masse på et elektron, som partikkelen i boksen


# la x og t være array
def psi_n(n,x):
    return np.sqrt(2/a)*np.sin(n*np.pi*x/a)

def E_n(n):
    return ha*np.pi*(n**2)/(2*m*a**2)

def Psi(f1,f2,x,t):
    psi_1 = f1(1,x); psi_3 = f1(3,x); psi_5 = f1(5,x)
    E1 = f2(1) ; E3 = f2(3); E5 = f2(5)
    im = np.complex(0,1)
    phi_1 = np.cos(-m/ha*E1*t) + im*np.sin(-m/ha*E1*t); phi_3 = np.cos(-m/ha*E3*t) + im*np.sin(-m/ha*E3*t);
    phi_5 = np.cos(-m/ha*E5*t) + im*np.sin(-m/ha*E5*t)
    return np.sqrt(2)/16*(psi_1*phi_1 - 5*psi_3*phi_3 + 10*psi_5*phi_5)


x = np.linspace(0,a,1000); t1 = 0.7 ; t2 = 1.4 #s

wave_t1 = Psi(psi_n,E_n,x,t1)
wave_t2 = Psi(psi_n,E_n,x,t2)
prob_t1 = np.conj(wave_t1)*wave_t1
prob_t2 = np.conj(wave_t2)*wave_t2

# imag og real kan benyttes for å hente ut den imaginære delen og reelle.

plt.figure(figsize = (10,6))
ax1 = plt.subplot(2,2,1)
ax1.set_title('TASL (t: 0.7s)')
ax1.plot(x,np.real(wave_t1),label = 'Re')
ax1.plot(x,np.imag(wave_t1), label = 'Im')
ax1.set(ylabel='psi')
ax1.legend()

ax2 = plt.subplot(2,2,2)
ax2.set_title('Sannsynlighetstetthet (t:0.7s)')
ax2.plot(x,prob_t1)
ax2.set(ylabel='$psi^{2}$')

ax3 = plt.subplot(2,2,3)
ax3.set_title('TASL (t:0.7s)')
ax3.plot(x,np.real(wave_t2),label = 'Re')
ax3.plot(x,np.imag(wave_t2), label = 'Im')
ax3.set(xlabel='x [nm]', ylabel='psi')
ax3.legend()

ax4 = plt.subplot(2,2,4)
ax4.set_title('Sannsynlighetstetthet (t:1.4s)')
ax4.plot(x,prob_t2)
ax4.set(xlabel='x [nm]', ylabel='$psi^{2}$')

plt.show()
#####################
