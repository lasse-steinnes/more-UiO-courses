import numpy as np
import matplotlib.pyplot as plt

## matematisk modell for sneglehuset ##
H = 0.67 # cm
l = 3.5 # cm
rho = 1 # g/m^3
c  = 1.43*10**5 # cm/s
m = 0.143 #g/cm^2
k = lambda x: 10**(8)*np.exp(-4*x) # g/cm^2 s^2 endre til gram

# lager en klasse for modellen
# eta er eta
# leapfrog metode

class cochlea_solver():
    def __init__(self,initial_psi,initial_prev_psi,k,dt,dx):
        # Beregner eta fra initial

        # definerer parametere
        # la prev være tidssteget tidligere t-dt, t psi eta, t +dt er new verdien
        self.prev_psi = np.zeros(initial_psi.shape); self.psi = np.zeros(initial_psi.shape); self.psi_new =np.zeros(initial_psi.shape)
        self.prev_eta = np.zeros(initial_psi.shape); self.eta = np.zeros(initial_psi.shape); self.eta_new = np.zeros(initial_psi.shape)
        self.dt = dt; self.dx = dx; self.k = k

        # lagrer initialbetingelser
        self.psi[:] = initial_psi[:]
        self.prev_psi[:] = initial_prev_psi[:]



    def advance(self):
        prev_psi = self.prev_psi; psi = self.psi; psi_new = self.psi_new
        prev_eta = self.prev_eta; eta = self.eta; eta_new = self.eta_new;
        dt = self.dt; self.dx = dx
        psi_new = np.zeros(psi.shape); eta_new = np.zeros(psi.shape)
        # har at psi er et todimensjonalt array, s.a. psi[1,2] f.eks
        # + 1 da blir index 2 først, -1 blir index -2 sist, slipper index for tida
        # alle må ha like lengde til slutt
        psi_new[1:-1] = (c*dt/dx)**2*(psi[2:] - 2*psi[1:-1] + psi[0:-2]) + 2*psi[1:-1] - prev_psi[1:-1]\
         - 2*rho*c**2/H*(dt**2)*(psi[1:-1]-k[1:-1]*eta[1:-1])/m

        eta_new[1:-1] = 1/m*(psi[1:-1]-k[1:-1]*eta[1:-1])*dt**2 + 2*eta[1:-1] - prev_eta[1:-1]

        prev_eta[:] = eta[:]; eta[:] = eta_new[:];
        prev_psi[:] = psi[:]; psi[:] = psi_new[:]

    def psi_solution(self):
        return self.psi
    def eta_solution(self):
        return self.eta

# Velg en trykkamplitude på 60 desibel (SPL)
# Velge grensebetingelser.

# benytte for løkke til slutt for å iterere gjennom mange tider?
# løser for en x -array
f = 1000
N = 100;  t_end = 10/f # antall sekunder
x = np.linspace(0,l,N);
dx = x[1]-x[0]; dt = 0.5*dx/c
k = k(x)
t = np.arange(0, t_end, dt)
Nt = len(t)
# lager array
psi = np.zeros((Nt,N)) #rad, kolonner
eta = np.zeros((Nt,N))


# Skal ha en trykkamplitude, som sprer seg ut langs basillarmembranen
p0 = 0.2 # pascal, sgs enheter
print(p0)
f = 1000     # Hz, frekvens midt i spekteret for det menneskelige øret.
omega = 2*np.pi*f
p = lambda t: p0*np.sin(omega*t)

# må ha to intitialbetingelser
psi[:,0] = p(t)

# Benytt trykkbølge for alle t langs x - dx for første iterasjon
cochlea = cochlea_solver(psi[1 ,:],psi[0,:],k,dt,dx)

for i in range(1,len(t)-1):                        # iterere over alle tidssteg
#    cochlea = cochlea_solver(psi[i ,:],psi[i-1,:],k,dt,dx)
    cochlea.advance()
    psi[i+1,:] = cochlea.psi_solution()
    eta[i+1,:] = cochlea.eta_solution()
    t[i+1] = t[i] + dt
    psi[i+1, 0] += p(t[i+1])

#print(eta)

plt.plot(x, eta[-1,:])
plt.show()


#
