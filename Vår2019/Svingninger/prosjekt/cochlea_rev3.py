# program med enkel p og finpussa plott
import numpy as np
import matplotlib.pyplot as plt

## matematisk modell for sneglehuset ##
H = 0.67 # cm
l = 3.5 # cm
rho = 1 # g/m^3
c  = 1.43*10**5 # cm/s
m = 0.143 #g/cm^2
k = lambda x: 4e9*np.exp(-4*x) # g/cm^2 s^2 endre til gram

# lager en klasse for modellen
# eta er eta
# leapfrog metode

class cochlea_solver():
    def __init__(self,Nt, N, initial_xi,xi_arr,eta_arr,k,dt,dx): # tar inn en array for xi ved x0 for alle t
        # Beregner eta fra initial
        # tar inn tomme array og ett med init

        # definerer parametere
        # la prev være tidssteget tidligere t-dt, t xi eta, t +dt er new verdien
        self.Nt = Nt; self.N = N ;
        self.xi = xi; self.eta = eta
        # lagrer initialbetingelser
        self.xi[:,0] = initial_xi
        self.dx = dx
        self.dt = dt


    def solve(self):
        xi = self.xi; eta = self.eta
        dt = self.dt; self.dx = dx;
        Nt = self.Nt; N = self.N

        for j in range(1,Nt-1): # j tid
            for i in range(1,N-1): # i posisjon
                xi[j+1,i] = (c*dt/dx)**2*(xi[j,i+1] - 2*xi[j,i] + xi[j,i-1]) + 2*xi[j,i] - xi[j-1,i]\
                 - 2*rho*c**2/H*(dt**2)*(xi[j,i]-k[i]*eta[j,i])/m

                eta[j+1,i] = 1/m*(xi[j,i]-k[i]*eta[j,i])*dt**2 + 2*eta[j,i] - eta[j-1,i]
        return xi,eta


# Velg en trykkamplitude på 60 desibel (SPL)
# Velge grensebetingelser.

# benytte for løkke til slutt for å iterere gjennom mange tider?
# løser for en x -array
f = 1000
N = 150;  t_end = 10/f # antall sekunder
x = np.linspace(0,l,N);
dx = x[1]-x[0]; dt = 0.5*dx/c #
k = k(x)
t = np.arange(0, t_end, dt)
Nt = len(t)

print('Antall punkter i tid',Nt)
print('Antall punkter i posisjon',N)
print('Lengde på tidsintervall',t_end,'sekunder')

xi = np.zeros((Nt,N)) #rad, kolonner
eta = np.zeros((Nt,N)) #


# Skal ha en trykkamplitude, som sprer seg ut langs basillarmembranen
p0 = 0.2 #sgs enheter (Ba) dyb/cm^2
print('Trykksamplituden',p0,'')
f = 1000     # Hz, frekvens midt i spekteret for det menneskelige øret.
omega = 2*np.pi*f
p = lambda t: p0*np.sin(omega*t)

# må ha to intitialbetingelser
init = np.zeros(Nt)
init = p(t)

# Benytt trykkbølge for alle t langs x - dx for første iterasjon
cochlea = cochlea_solver(Nt,N,init,xi,eta,k,dt,dx)
xi, eta = cochlea.solve()

# Finner x-posisjonen der utslaget er størst, og markerer dette
# finn x-posisjonen med greenwood funksjonen:


print(max(eta[-1,:]))
max_in = np.where(eta[-1,:] == max(eta[-1,:]))
print('maxindex x',max_in)

# plotter for max_indexen over alle tider # markerer tida der resonansen oppstår
maxt_in = np.where(eta[:,max_in] == max(eta[:,max_in]))
print(maxt_in)



# plotte for ulike tider der resonansen forekommer
plt.figure()
plt.plot(x, eta[-1,:],label = '{:.3e} s'.format(t[-1]))
plt.plot(x,eta[100,:],label = '{:.3e}s'.format(t[100]))
plt.plot(x,eta[maxt_in[0][0]],label = '{:.3e}s'.format(t[maxt_in[0][0]]))
plt.axvline(x = x[max_in[0][0]],color = 'r', linewidth = 0.5,label ='x:{:.3e}'.format(x[max_in[0][0]]))
plt.xlabel('Posisjon langs basilarmembranen [cm]')
plt.ylabel('Amplitude [cm]')
plt.legend()
plt.show()

# resonansen blir stående


# plotter utslag over tid for posisjonen der  amplituden er max
plt.figure()
plt.plot(t, eta[:,max_in[0]])
plt.axvline(x = t[maxt_in[0][0]],color = 'r', linewidth = 0.5,label ='t:{:.3e}'.format(t[maxt_in[0][0]]))
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [cm]')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.sqrt(k/m)*1/(2*np.pi),(eta[-1,:])) # plotter mot egenfrekvensen
plt.xlabel('Frekvens [Hz]')
plt.ylabel('Amplitude [cm]')
plt.show()

# plotter trykksforskjellen langs membranen for ulike tidspunkt
plt.figure()
plt.plot(x,xi[-1,:],'--',label = '{:.3e} s'.format(t[-1]))
plt.plot(x,xi[100,:], label = '{:.3e}s'.format(t[100]))
plt.plot(x,xi[maxt_in[0][0],:], label = '{:.3e}s'.format(t[maxt_in[0][0]]))
plt.xlabel('Posisjon langs basilarmembranen [cm]')
plt.ylabel('Trykksforskjellen, p - q [Ba]')
plt.legend()
plt.show()
