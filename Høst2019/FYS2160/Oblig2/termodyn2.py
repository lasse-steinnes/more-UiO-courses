### Program for termodynamic assignment 2 ###
import numpy as np
import matplotlib.pyplot as plt

##  3.3
n = 1000
V = np.linspace(0.4,6,n)
T = np.arange(0.8,1.05,step = 0.05)

print(T)

def P_hat(T,V):
    return 8*T/(3*V-1)-3/(V**2)

plt.figure()
plt.title('3.3: Pressure ($\hat{P}$) as a function of volume ($\hat{V}$ (dimensionless)')
for i in range(len(T)):
    plt.plot(V,P_hat(T[i],V), label ='T:{:.2f}'.format(T[i]))

plt.xlabel('$\hat{V}$')
plt.ylabel('$\hat{P}$')
plt.legend()
plt.show()

## 3.5
rho = np.linspace(0.0,2.0,n)

def P_hat2(T,rho):
    return 8*rho*T/(3-rho) - 3*rho**2

plt.figure()
plt.title('3.5: Pressure ($\hat{P}$) as a function of density ($\hat{rho}$ (dimensionless)')
for i in range(len(T)):
    plt.plot(rho,P_hat2(T[i],rho), label ='T:{:.2f}'.format(T[i]))

plt.xlabel('$\hat{rho}$')
plt.ylabel('$\hat{P}$')
plt.legend()
plt.show()

## 3.6 For nitrogen, PV isotherms
Pc = 33.6 # atm
Vc = 0.089 # l/mol
Tc = 126 # K
T = (1/Tc)*np.array([77,100,110,115,120,125])

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

plt.figure()
plt.title('3.6: PV istotherm for Nitrogen ($N_{2}$)')

for i in range(len(T)):
    P = P_hat(T[i],V)*Pc
    plt.plot(V*Vc,P,label ='T:{:.2f}'.format(T[i]*Tc))

plt.xlabel('${V}$')
plt.ylabel('${P}$')
plt.text(2, 0.65, r'Pc = 33.6 atm, Vc = 0.089 l/mol,Tc = 126 K', fontdict=font)
plt.legend()
plt.show()

## 3.7.
# Vg and Vl for different Ts 100,110,115,120,125
Vl = np.array([0.045,0.05,0.055,0.06,0.075])
Vg = np.array([0.4,0.245,0.20,0.152,0.1])

P_ = np.array([12,19,22.5,27.5,32.55])
plt.figure()
plt.figure('3.7: Finding Vl and Vg using Maxwell equal area construction')
for i in range(len(T)):
    P = P_hat(T[i],V)*Pc
    plt.plot(V*Vc,P,label ='T:{:.2f}'.format(T[i]*Tc))
for i in range(len(P_)):
    plt.plot(V*Vc,np.zeros(len(V*Vc)) + P_[i],'--')

plt.xlabel('${V}$')
plt.ylabel('${P}$')
plt.legend()
plt.show()


plt.figure()
plt.title('3.7: Finding $T_{c}$ for Nitrogen using Vg, Vl')
plt.plot(T,Vg-Vl)
plt.xlabel('T')
plt.ylabel('$V_{g}-V_{l}$')
plt.show()
