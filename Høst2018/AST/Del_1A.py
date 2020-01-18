### AST-oppgaver DEL 1A
import numpy as np


## Oppgave 1A.6 Innlevering
#
import numpy as np

from AST2000SolarSystemViewer import AST2000SolarSystemViewer
seed = 61252; system = AST2000SolarSystemViewer(seed)

myp_radius = system.radius[0]; myp_mass = system.mass[0]
print('Masse til planeten:{:.4e}'.format(myp_mass))

# 1: Escape velocity from home planet
G = 4 * np.pi * np.pi   # Gravitasjonskonstant i astronomiske enheter
M = myp_mass;           # [solare masser]
R = myp_radius*1/1000;  # [m]
N = 10**5               # Antall partikler
L = 10**(-6);           # [m] Lengde til boksen
T = 10**4;              # [K]
                        # [eV/K] Boltzmann konstant
k = 1.380*10**(-23)    # [J/K]
m = 2/((6.022*10**23)*1000); # [kg]
sigma = np.sqrt(k*T/m)

# m = massen til atomene i gassen.

V_esc = np.sqrt((2*G*M)/R)
print( 'Hastigheten for å unnslippe planeten G-felt:{:}'.format(V_esc))
# 2 Simulering: Fordeling av gasspartiklene
# array med (x,y,z) uniform fordeling
np.random.seed(11); pos = np.random.uniform(0, L, size = (3,N)) #x_low, x_upper

# array med (vx, vy, vz) Gaussfordeling
np.random.seed(11); vel = np.random.normal(0,sigma, size = (3,N)) #(my, sigma)
#print(vel)
# må indeksere [rad][kolonne] Hver rad tilsvarer henholdsvis x,y,z

# 3
# a) Sjekk at gjennomsnittlig kinetisk energi er 3/2*kT
kE = (1/2)*m*(vel[0][:]**2 + vel[1][:]**2 + vel[0][:]**2)
mu_kE = np.mean(kE)
print('Numerisk gj. snitt for kinetisk energi, {:.4e}, er lik analytisk {:}'.format(mu_kE, 3/2*k*T))

# Omtrent


# b) At gjennomsnittlig absoluttverdi v av gasspartiklene følger uttrykk fra 1.A5
v_mu_an = np.sqrt((8*T*k)/(np.pi*m))

v_abs = np.sqrt(vel[0,:]**2 + vel[1,:]**2 + vel[2,:]**2)
v_mean = np.mean(v_abs)
print('Numerisk gj. snitt for absoluttverdi til hastigheten, {:.4e}, er lik analytisk {:.4e}'.format(v_mean,v_mu_an))

# 4: Sjekk partikler som treffer veggen
t = np.zeros(1000)
t_tot = 10**(-9); step = 1001; dt = t_tot/(step)
"""
for i in range(0,999):
    pos = pos + vel*dt
    collision_points = np.logical_or(pos == L,pos > L)
    collisions_indices= np.where(collision_points == True)
    pos[collisions_indices] = L - 10**5
    vel[collisions_indices] = - vel[collisions_indices]
    t[i+1] = t[i] + dt
"""
# Oppgave 5. For en vegg
# Skal finne trykket: P = F/A, og F = dp/dt,
t1 = np.zeros(1000)
col_sum = 0
px_sum = 0

for i in range(0,999):
    pos[0,:] = pos[0,:] + dt*vel[0,:]
    collision_points = np.logical_or(pos[0,:] > L,pos[0,:] == L)
    collisions_indices= np.where(collision_points == True)
    col_sum = col_sum + int(len(collisions_indices[0]))
    px = m*vel[0][collisions_indices[0]]
    px_sum = px_sum + np.sum(px)
    vel[0][collisions_indices] = - vel[0][collisions_indices[0]]
    t1[i+1] = t1[i] + dt

print('Antall kollisjoner:{:.4e}'.format(col_sum)) # 4679 antall kollisjoner for alle
            # 1524 for x-veggen.
#print(collision_points)
#print(collisions_indices)
#print(col_nr)
#print(np.shape(pos))
# Bevegelsesmengde: mv

print('Bevegelsesmengde:{:.4e}'.format(px_sum))

# f = 2px/dt
force = 2*px_sum/t_tot;A = L*L; P_num = force/A
print('Det numeriske trykket er:{:.4e}'.format(P_num))

# Analytisk
V = L**3; n = N/V; P_an = n*k*T
print('Det analytiske trykket er:{:.4e}'.format(P_an))

# Oppgave 6.6
# Lage hull, telle antall partikler, bevegelsesmengde
esc_sum = 0
px_sum = 0
test = 3

for i in range(0,999):
    pos[0,:] = pos[0,:] + dt*vel[0,:]
    escape_points = np.logical_and(pos[0,:] >= L,pos[1,:] <= L/2,pos[2,:] <= L/2)
    escape_indices= np.where(collision_points == True)
    esc_sum = esc_sum + int(len(escape_indices[0]))
    px = m*vel[0][escape_indices[0]]
    px_sum = px_sum + np.sum(px)
    pos[0][escape_indices] = 0
    t1[i+1] = t1[i] + dt

print('Antall partikler som unnslipper:{:.4e}'.format(esc_sum))
print('Bevegelsesmengde fra ett gasskammer: {:.4e}'.format(px_sum))

## 7, 8, 9 ##
