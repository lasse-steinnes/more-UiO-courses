######## Oblig2 ########
import numpy as np
import matplotlib.pyplot as plt
"""Oppgave 3.3"""
# c) Skal modellere membranens egenfrekvens som en rekke
# harmoniske oscillatorer

# fra tykk til tynn
# Den laveste/høyeste frekvens som øret kan høre
# Deler inn i 3000 deler, jamfør 3000 nerveceller.
length = np.linspace(0,30e-3,3000)
punktle = length[1]
width = np.linspace(0.3e-3,0.1e-3,3000)
height = np.linspace(0.1e-3,0.3e-3,3000)
rho = np.linspace(1.5e3,2.5e3,3000)
k = np.linspace(10**(-1),10**(-6),3000)


m = punktle*width*height*rho

f = np.sqrt(k/m)/(2*np.pi)
print('minimumsfrekvens:', np.min(f))
print('maksimumfrekvens',np.max(f))

plt.plot(length,f)
plt.xlabel('lengde [m]')
plt.ylabel('Frekvens [Hz]')
plt.show()

### Oppgave d ###
# Vet at amplituden til partikulærløsningen er gitt ved
f01 = 261.63 ; f02 = 277.18; faktor = 2*np.pi;
b = 2e-8

A1 = 1/np.sqrt(m*(((f*faktor)**2-(f01*faktor)**2)**2 + (b*(f01*faktor)/m)**2))
A2 = 1/np.sqrt(m*(((f*faktor)**2-(f02*faktor)**2)**2 + (b*(f02*faktor)/m)**2))

plt.plot(length[2900:2970],A1[2900:2970])
plt.plot(length[2900:2970],A2[2900:2970])
plt.xlabel('lengde [m]')
plt.ylabel('Amplitude')
plt.show()

# oppgave e)
delta_f = (277.18-261.63)
f = 261.63
Q = f/delta_f
print('Q-verdien til systemet er',Q)

# oppgave f)
# hvor lenge den svinger etter ekstern stimulans slutter
delta_t = 1/((2*np.pi)*delta_f)
print('stoppetida er',delta_t)

""" Kjøreeksempel
python oblig2fys2130.py

minimumsfrekvens: 5.81054816446
maksimumfrekvens 2372.1463548
Q-verdien til systemet er 16.82508038585208
stoppetida er 0.01023504457182606
"""



########################
