# Skal plotte bølgefunksjonen for en partikkel
# potensial tilsvarende endelig brønn
import numpy as np
import matplotlib.pyplot as plt

# konstanter
m = 939.6 #[MeV/c^2], massen til nøytronet.
ha = 6.58*10**(-16) #  [eV s], plancks reduserte konstant
r0 = 1.25*10**(-15) # [m], grunnradius
nuc = 235 # nukleontallet til uranium-235
R = r0*(nuc)**(1/3)*10**9# [nm] radius til kjernen i et stabilt atom
a = R
V0 = 10 # [MeV], potensialdybden
E1 = 0.5*10; E2 = 3.0# [MeV], energitilstander

# Skal lime sammen bølgefunksjonen for de ulike delene av brønnen

class TUSL_finitewell():
    def __init__(self,E,a,m,V0):
        self.k = np.sqrt(2*m*E)/ha; self.a = a; self.l = np.sqrt(2*m*(E+V0))/ha
        k = self.k; a = self.a; l = self.l
#bølgekonstanter
        self.im = np.complex(0,1); im = self.im
        self.A = 1; A = self.A
        self.F = np.exp(-2*im*k*a)*A/(np.cos(2*l*a)-im*(k**2+l**2)/(2*k*l)*np.sin(2*l*a))
        F = self.F
        self.B = im*np.sin(2*l*a)/(2*k*l)*(l**2 - k**2)*F
        self.C = (np.sin(l*a) + im*k/l*np.cos(l*a))*F*np.exp(im*k*a)
        self.D = (np.cos(l*a)- im*k/l*np.sin(l*a))*F*np.exp(im*k*a)

    def psi_left(self,x):
        im = self.im; A = self.A; B = self.B; k = self.k
        return A*np.exp(im*k*x) + B*np.exp(-im*k*x)

    def psi_mid(self,x):
        im = self.im; C = self.C; D = self.D; l = self.l
        return C*np.sin(l*x) + D*np.cos(l*x)

    def psi_right(self,x):
        im = self.im; F = self.F, k = self.k
        return F*np.exp(im*k*x)


width = 30*10**(-15)*10**9 # [nm] halvebredde plott
n = 130
left = np.linspace(-width,-a-a/n,n)
mid = np.linspace(-a+a/n,a-a/n,n)
right = np.linspace(a+a/n,width,n)

# instanse
wave11 = TUSL_finitewell(E1,a,m,V0)
wave12 = TUSL_finitewell(E1,a,m,0)
# bølgefunksjoner
psi_l1 = wave12.psi_left(left); psi_m1 = wave11.psi_left(mid)
psi_r1 = wave12.psi_left(right)

# instanse for andre energitilstand
wave21 = TUSL_finitewell(E2,a,m,V0)
wave22 = TUSL_finitewell(E2,a,m,0)

psi_l2 = wave22.psi_left(left); psi_m2 = wave21.psi_left(mid)
psi_r2 = wave22.psi_left(right)

# sannsynlighet np.conj
prob_l1 = np.conj(psi_l1)*psi_l1; prob_m1 = np.conj(psi_m1)*psi_m1
prob_r1 = np.conj(psi_r1)*psi_r1

prob_l2 = np.conj(psi_l2)*psi_l2; prob_m2 = np.conj(psi_m2)*psi_m2
prob_r2 = np.conj(psi_r2)*psi_r2

# Må plotte potensialfunksjonen; np real og np imag
# Sannsynlighetstetthet.# np real og np imag
# Alt mellom -30 til 30 fm,

plt.figure(figsize = (10,6))
plt.title('TUSL (E:0.5 MeV)')
plt.plot(left,np.real(psi_l1),'dodgerblue',label = 'Re')
plt.plot(left,np.imag(psi_l1),'darkorange', label = 'Im')
plt.plot(mid,np.real(psi_m1),'dodgerblue')
plt.plot(mid,np.imag(psi_m1),'darkorange')
plt.plot(right,np.real(psi_r1),'dodgerblue')
plt.plot(right,np.imag(psi_r1),'darkorange')
plt.xlabel('x [nm]')
plt.ylabel('psi')
plt.legend()
plt.show()

plt.figure(figsize = (10,6))
plt.title('Sannsynlighetstetthet (E:0.5 MeV)')
plt.plot(left,prob_l1,'dimgray')
plt.plot(mid,prob_m1,'dimgray')
plt.plot(right,prob_r1,'dimgray')
plt.xlabel('x [nm]')
plt.ylabel('$|psi|^{2}$')
plt.show()

plt.figure(figsize = (10,6))
plt.title('TUSL (E:3.0 MeV)')
plt.plot(left,np.real(psi_l2),'dodgerblue',label = 'Re')
plt.plot(left,np.imag(psi_l2),'darkorange', label = 'Im')
plt.plot(mid,np.real(psi_m2),'dodgerblue')
plt.plot(mid,np.imag(psi_m2),'darkorange')
plt.plot(right,np.real(psi_r2),'dodgerblue')
plt.plot(right,np.imag(psi_r2),'darkorange')
plt.xlabel('x [nm]')
plt.ylabel('psi')
plt.legend()
plt.show()

plt.figure(figsize = (10,6))
plt.title('Sannsynlighetstetthet (E:3.0 MeV)')
plt.plot(left,prob_l2,'dimgray')
plt.plot(mid,prob_m2,'dimgray')
plt.plot(right,prob_r2,'dimgray')
plt.xlabel('x [nm]')
plt.ylabel('|$psi|^{2}$')
plt.show()

#####################
