### AST-oppgaver DEL 1A
import numpy as np


## Oppgave 1A.6 Innlevering
#
import numpy as np

from AST2000SolarSystemViewer import AST2000SolarSystemViewer
seed = 61252; system = AST2000SolarSystemViewer(seed)

myp_radius = system.radius[0]; myp_mass = system.mass[0]

# 1: Escape velocity from home planet
G = 4 * np.pi * np.pi   # Gravitasjonskonstant i astronomiske enheter
M = myp_mass*2*10**30;           # [solare masser] -> kg
G2 = 6.6*10**(-11)
R = myp_radius*1000;  # [m]
N = 10**5               # Antall partikler
L = 10**(-6);           # [m] Lengde til boksen
T = 10**4;              # [K]
                        # [eV/K] Boltzmann konstant
k = 1.380*10**(-23)    # [J/K]
m = 2/((6.022*10**23)*1000); # [kg]
m_sat = 1000                # [kg]


print('Massen til planeten: {:} (kg)'.format(M))
print('Radius til planet: {:.4e} (m)'.format(R))
print('massen til en partikkel {:} kg'.format(m))
sigma = np.sqrt(k*T/m)

# m = massen til atomene i gassen.
print('Oppgave 1:')
print('---------------')

V_esc = np.sqrt((2*G2*M)/R)
print( 'Hastigheten for å unnslippe planeten G-felt:{:}'.format(V_esc))
# 2 Simulering: Fordeling av gasspartiklene
# array med (x,y,z) uniform fordeling
np.random.seed(11); pos = np.random.uniform(0, L, size = (3,N)) #x_low, x_upper

# array med (vx, vy, vz) Gaussfordeling
np.random.seed(11); vel = np.random.normal(0,sigma, size = (3,N)) #(my, sigma)
#print(vel)
# må indeksere [rad][kolonne] Hver rad tilsvarer henholdsvis x,y,z

# 3
print('Oppgave 3:')
print('---------------')
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
print('Oppgave 4')
print('---------------')
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
print('Oppgave 5')
print('---------------')
t1 = np.zeros(1000)
col_sum = 0
px_sum = 0
F_tot = 0


for i in range(0,999):
    posfør = pos[0,:]
    pos[0,:] = pos[0,:] + dt*vel[0,:]
    collision_points = np.logical_or(pos[0,:] > L,pos[0,:] == L)
    collisions_indices= np.where(collision_points == True)
    col_sum = col_sum + int(len(collisions_indices[0]))
    px = m*vel[0][collisions_indices[0]]
    px_sum = px_sum + np.sum(px)
    F_tot = F_tot + 2*np.sum(px)/dt
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
force = 2*px_sum/t_tot;A = L*L; P_num = F_tot/A;
print('Det numeriske trykket er:{:.4e}'.format(P_num))

# Analytisk
V = L**3; n = N/V; P_an = n*k*T
print('Det analytiske trykket er:{:.4e}'.format(P_an))

# Oppgave 6.6
# Lage hull, telle antall partikler, bevegelsesmengde
print('Oppgave 6')
print('---------------')
esc_sum = 0
px_sum = 0
test = 3

for i in range(0,999):
    pos[0,:] = pos[0,:] + dt*vel[0,:]
    collision_points = np.logical_and(pos[0,:] > L,pos[1,:]> L/2, pos[2,:] > L/2)
    collisions_indices= np.where(collision_points == True)
    escape_points = np.logical_and(pos[0,:] >= L,pos[1,:] <= L/2,pos[2,:] <= L/2)
    escape_indices= np.where(escape_points == True)
    esc_sum = esc_sum + int(len(escape_indices[0]))
    px = m*vel[0][escape_indices[0]]
    px_sum = px_sum + np.sum(px)
    pos[0][escape_indices] = 0
    vel[0][collisions_indices] = - vel[0][collisions_indices[0]]
    t1[i+1] = t1[i] + dt

print('Antall partikler som unnslipper:{:.4e}'.format(esc_sum))
print('Bevegelsesmengde fra ett gasskammer: {:.4e}'.format(px_sum))

## 7:
# siden bevegelsesmengde er p = mv, må v = p/m, s.a.
# siden bevegelsesmengde er bevart må boksen skyte i motsatt retning med delta v
print('Oppgave 7')
dv =  px_sum/m_sat
print('Endring i hastighet etter t: {:} er {:} m/s'.format(t_tot, dv))

"""
Endring i hastighet etter t: 1e-09 er 1.2524413363188593e-24 m/s
"""

## 8:
# Skal få rakett til escape vel innen 20 min. Hvor mange bokser trengs?
#Hastigheten for å unnslippe planeten G-felt:9766.400672091866 m/s
v_tot = 0    # initiell hastighet fra planetens
sek_tot = 60*20
vel_shift_enboks = dv*sek_tot
num_boks = V_esc/vel_shift_enboks
total_v = vel_shift_enboks*num_boks
print('For å oppnå en hastighet {:} m/s ila. {:} sekm, trengs {:} bokser'.format(total_v,sek_tot,num_boks))

## 9:
# N antall partikler i hver boks, har at massen til en partikkel er m, s.a.
print('Oppgave 9')
m_drivstoff = N*m*num_boks
print('Den totale massen drivstoff som trengs for å unnslippe er {:} kg'.format(m_drivstoff))


##############
"""
Kjøreeksempel:
python Del_1Arev.py
Massen til planeten: 3.8879423624027683e+24 (kg)
Radius til planet: 5.3805e+06 (m)
massen til en partikkel 3.321155762205248e-27 kg
Oppgave 1:
---------------
Hastigheten for å unnslippe planeten G-felt:9766.400672091866
Oppgave 3:
---------------
Numerisk gj. snitt for kinetisk energi, 2.0653e-19, er lik analytisk 2.0699999999999996e-19
Numerisk gj. snitt for absoluttverdi til hastigheten, 1.0284e+04, er lik analytisk 1.0286e+04
Oppgave 4
---------------
Oppgave 5
---------------
Antall kollisjoner:4.7177e+04
Bevegelsesmengde:8.5762e-19
Det numeriske trykket er:1.7169e+06
Det analytiske trykket er:1.3800e+04
Oppgave 6
---------------
Antall partikler som unnslipper:7.4900e+02
Bevegelsesmengde fra ett gasskammer: 1.2524e-21
Oppgave 7
Endring i hastighet etter t: 1e-09 er 1.2524413363188593e-24 m/s
For å oppnå en hastighet 9766.400672091866 m/s ila. 1200 sekm, trengs 6.498242265513341e+24 bokser
Oppgave 9
Den totale massen drivstoff som trengs for å unnslippe er 2158.167474431532 kg
"""
