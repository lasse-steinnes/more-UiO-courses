### lab 1 ###

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

mikro = 10**(-6)
milli = 10**(-3)


v_h = np.array([1.1,6.2,11.7,17.3,23,28.6])
b_arr = np.array([6,42,74,108,142,175])
b_new = np.linspace(0,180,100) # en serie punkter
n = 1# tilpass til et n-te gradspolynom

coefs = poly.polyfit(b_arr, v_h, n)
print('stigningstall',coefs)
ffit = poly.polyval(b_new, coefs)

plt.plot(b_new, ffit,'-')
plt.plot(b_arr,v_h,'o')
plt.xlabel('mT')
plt.ylabel('mV')
plt.show()
"""
# ØNsker å beregne logaritmisk skala
"""
U = np.array([8.1,6.43,5.0,3.9,3.1,2.4,1.95,1.55,1.2,0.97,0.77,0.61,0.48,0.38,0.31,0.204,0.191,0.151,0.121])
Ulog = np.log(U)
t = np.linspace(0,360, 360/19 + 1)
n = 1

t_new = np.linspace(0,360,1000)
#print(t,Ulog)

coefs = poly.polyfit(t, Ulog, n)
print('stigningstall',coefs)
ffit = poly.polyval(t_new, coefs)

plt.plot(t_new, ffit,'-')
plt.plot(t,Ulog,'o')
plt.xlabel('t')
plt.ylabel('Ulog')
plt.show()

## Finner tau
tau = -1/coefs[1]
print(tau)

## Skal finne indre resistans
C = 8.3*mikro #farad
R = tau/C
print('indre resistans',R,'ohm')

### Lager array
I = np.array([20.019,14.52,10.289,8.61,6.921])
V = np.array([0.5656,0.4098,0.2901,0.24275,0.1951])
n = 1
I_new = np.linspace(5,25,1000)

coefs = poly.polyfit(I, V, n)
print('stigningstall',coefs)
ffit = poly.polyval(I_new, coefs)

plt.plot(I_new, ffit,'-')
plt.plot(I,V,'o')
plt.xlabel('I')
plt.ylabel('V')
plt.show()

print('indre resistans ampermeter',coefs[1],'ohm')
# curvefit fra scipy
##### Lager plot og beregner I

R = np.array([1,1.5,2.5,4,10])
V = np.array([42.2,56.0,80.15,106.3,159.34])*milli
I = V/R
print(I)

n = 1
I_new = np.linspace(0.00001,0.05,1000)

coefs = poly.polyfit(I, V, n)
print('stigningstall',coefs)
ffit = poly.polyval(I_new, coefs)

plt.plot(I_new, ffit,'-')
plt.plot(I,V,'o')
plt.xlabel('I')
plt.ylabel('V')
plt.show()

# emf: 0.22899456 V  # Ri: 4.54881367] Ohm
# avhenger av temp



# Oppgave 4
# lager array for R
V4_kobber = np.array([0.05,0.04])*milli
A4_kobber =np.array([1.5,1.0])
A4_al =np.array([1.502,1.001])
V4_al =np.array([0.09,0.06])*milli

V2_kobber =np.array([4.95,3.24])*milli
A2_kobber =np.array([1.503,1.003])
V2_al =np.array([6.59,4.30])*milli
A2_al =np.array([1.504,1.001])

R4_al = np.zeros(2)
R4_kob =np.zeros(2)
R2_al =np.zeros(2)
R2_kob =np.zeros(2)

for i in range(2):
    R4_al[i] = V4_al[i]/A4_al[i]
    R4_kob[i] =V4_kobber[i]/A4_kobber[i]
    R2_al[i] =V2_al[i]/A2_al[i]
    R2_kob[i] =V2_kobber[i]/A2_kobber[i]

print('resistans 4-kobling kobber,al:', R4_kob,R4_al)
print('resistans 2-kobling kobber al,', R2_kob,R2_al)
##
print('Feilkoeffisient for R2/R4 kobber, al',R2_kob/R4_kob,R2_al/R4_al)

####
####
e0 = np.array([3.78,4.14,4.72,5.46,4.43,5.09,5.15,5.21])*milli
omega = np.array([8.51,9.24,10.5,11.9,9.78,11.1,11.4,11.6])

def mean(e0,omega):
    NA = 11 #m**2
    B = np.zeros(len(e0))
    for i in range(len(e0)):
        B[i] = e0[i]/(omega[i]*NA)
    mean_B = sum(B)/len(e0)
    return mean_B

B_mean = mean(e0,omega)
print('Gjennomsnittlig B-felt',B_mean,'T')
