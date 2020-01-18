### Ukesoppgave 4: Numeriske metoder ###
import numpy as np
import matplotlib.pyplot as plt

## Lage sin egen RK4 - løser ##
def RK4(xn,vn,tn,dt,u): # u er funksjonen for den andrederiverte
    ## bestemmer iterasjonvariable
    half_delta_t = 0.5*dt; t_p_half = tn + half_delta_t; tp = tn + dt
    ## Første ledd
    x1 = xn; v1 = vn; a1 = u(xn,vn,tn)
    # andre ledd
    x2 = x1 + v1*half_delta_t; v2 = v1 + a1*half_delta_t; a2 = u(x2,v2,t_p_half)
    # tredje
    x3 = x1 + v2*half_delta_t; v3 = v1 + a2*half_delta_t; a3 = u(x3,v3,t_p_half)
    # fjerde ledd
    x4 = x1 + v3*dt; v4 = v1 + a3*dt; a4 = u(x4,v4,tp)
    # returnerer som gjennomsnitt over disse
    dt_6 = dt/6; xp = xn + dt_6*(v1 + 2*(v2+v3)+v4)
    vp = vn + dt_6*(a1 + 2*(a2+a3)+a4);
    return [xp, vp, tp]

def u(xn,vn,tn):
    return -k/m*xn -b/m*vn

def intloop(T,dt,z0,u):
    N  = int(T/dt); t = np.zeros(N)
    z  = np.zeros((N, 3))
    z[0]  = z0
                        # z0 er her en vektor
    for i in range(0,N-1):
        z[i+1] = RK4(z[i,0],z[i,1],t[i],dt,u)
        t[i+1] = t[i]+dt
    return z

# initialbetingelser
m = 0.10; k = 10; b = 0.10; T = 15; dt = 0.01
z = intloop(T,dt,[0.1,0,0],u)


# analytisk for underkritisk dempet svingning
def z_an(t):
    gamma = b/(2*m)
    omega = np.sqrt(k/m)
    omega_prime = np.sqrt(omega**2-gamma**2)
    return 0.1*np.exp(-gamma*t)*np.cos((omega_prime)*t)

# passende tidssteg?
# samsvar numerisk og analytisk? Ja, så lenge tidssteget er lite nok
plt.plot(z[:,2],z[:,0],label = 'Numerisk RK4')
plt.plot(z[:,2],z_an(z[:,2]),'--',label = 'Analytisk løsning')
plt.legend()
plt.xlabel('tid [s]')
plt.ylabel('Utslag [m]')
plt.show()


# Kritisk dempning: endret initialbetingelser
# gamma = b/2*m = sqrt(k/m) = omega, redefinerer k
b = 2*np.sqrt(k*m); gamma = b/(2*m); omega = np.sqrt(k/m)
print(gamma,omega)

z_critical = intloop(T,dt,[0.1,0,0],u)


# overkritiskdempet gamma > omega
b = 8*2*np.sqrt(k*m)
z_overdamping = intloop(T,dt,[0.1,0,0],u)

plt.plot(z[:,2],z[:,0],label = 'subcritical')
plt.plot(z_critical[:,2],z_critical[:,0],label = 'critical')
plt.plot(z_overdamping[:,2],z_overdamping[:,0],label = 'supercritical')
plt.legend()
plt.xlabel('tid [s]')
plt.ylabel('Utslag [m]')
plt.show()


# initialbetingelser
m = 0.1; k = 10; b = 0.04; F = 0.1; T = 50; dt = 0.01
def u_tvungen(xn,vn,tn):
    omega_f =  4 # ikke amplituderesonans!!! #np.sqrt(k/m - (b**2)/(2*m**2)) # amplituderesonans
    return (F/m)*np.cos(omega_f*tn)-b/m*vn -k/m*xn


zf = intloop(T,dt,[0.1,0,0],u_tvungen)

plt.plot(zf[:,2],zf[:,0],label = 'Tvungen svingning')
plt.legend()
plt.xlabel('tid [s]')
plt.ylabel('Utslag [m]')
plt.tight_layout()
plt.show()

# d) Sjekke at frekvensresponsen
def u_tvungen(xn,vn,tn):
    return F/m*np.cos(omega_f*tn)-b/m*vn -k/m*xn

omegas = np.linspace(5, 15, 20)
As = []
plt.figure(figsize=(6, 4))
for omega_f in omegas:
    z = intloop(50, 0.1, [0,0,0],u_tvungen)
    utdrag = z[-int(len(z)/3):,0]
    A = (np.max(utdrag)-np.min(utdrag))/2
    As.append(A)

plt.figure(figsize=(6, 4))
plt.plot(omegas/2/np.pi, As)
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Amplitude (m)")
plt.tight_layout()
plt.show()

# Bruke ODE-løser fra en pakke til
# å løse IVP-er (initialverdiproblemer) istedenfor å skrive selv

from scipy.integrate import solve_ivp
"""
## a) Skal skrive om til system av første ordens ODEs
u0 = [0.1,0] # x0,v0
t_int = [0,10]

def du_dt(t,u):
    k = 10; b = 0.1; m = 0.1
    return np.array([u[1],-b/m*u[1]-k/m*u[0]])

## b) Løser IVPen med scipy
solution = solve_ivp(du_dt,t_int,u0)
plt.plot(solution.t,solution.y[0,:])
plt.show()
"""
"""
## c) Tvungne svingninger og påtrykte frekvenser,
# kvalitetsfaktor Q, to måter å beregne denne på
#og frekvenskurven (amplitude vs frekvens (frekvensrespons)),
def du_dt_tvungen(t,u):
    return [u[1],-b/m*u[1] -k/m*u[0] + F/m*np.cos(omega*t)]

N = 50; k = 10; m=0.1; b=0.1; F = 1
omega_values = np.linspace(3, 20, N)
A_values = np.zeros(N)

u0 = [0.1,0]

plt.figure()

for i in range(N):
    omega = omega_values[i]
    solution = solve_ivp(du_dt_tvungen, [0, 50], u0)
    # take out last half of solution
    utdrag_x = solution.y[0,-int(len(solution.t)/2):]
    A = (np.max(utdrag_x)-np.min(utdrag_x))/2
    A_values[i] = A

f_analytic = np.sqrt(k/m)/(2*np.pi) # Energien høyest ved frekvensresonans
Q_analytic = np.sqrt(m*k/b**2)
delta_f = f_analytic/Q_analytic

plt.plot(omega_values/(2*np.pi), A_values*A_values)
plt.axvline(f_analytic+delta_f/2, linestyle='--', color='k')
plt.axvline(f_analytic-delta_f/2, linestyle='--', color='k')
plt.xlabel("$f$ (Hz)")
plt.ylabel("$A^2$ (m$^2$) $\propto E$")
plt.show()
##
"""
