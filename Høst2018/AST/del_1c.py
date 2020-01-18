#########
# Benytt skjelettkode eller litt av den
#########

import numpy as np
import matplotlib.pyplot as plt

from AST2000SolarSystemViewer import AST2000SolarSystemViewer
seed = 61252; system = AST2000SolarSystemViewer(seed)
# 52

import numpy as np
import matplotlib.pyplot as plt


lambda0 = 656.3                 # Wavelenght for H_2 molecules, [nm]
light_speed = 2.99792458*10**8  # The speed of light, [m/s]
solar_mass = 1.989*10**30       # Mass of the sun/one solar mass, [kg]
G = 6.67408*10**(-11)           # Gravitational constant, [m^3/kg/s^2]
m_star = np.array([3.08,5.33,1.61,1.29,1.40])


def star_values(filee):
    """
    Function for putting stored values of time, wavelenght and flux, for a star
    in arrays. This makes it easy to compute with the values.
    Takes the file with stored star values (filee) as argument.
    """
    infile = open(filee,"r")    # Making it possible to read from the file
    t=[]                        #prepare lists
    lam=[]
    f=[]
    for line in infile:         # Looping over the lines in the file
        t_in,lam_in,f_in = line.split() # Splitting the line and putting the values in there one array
        t.append(t_in)                  #add to list
        lam.append(lam_in)
        f.append(f_in)

    t = np.array(t,dtype='float64')     #convert list to numpy array
    lam = np.array(lam,dtype='float64')
    f = np.array(f,dtype='float64')

    infile.close()              # Closing the file
    return t,lam,f              # Returning the value arrays

filee = ['star0_3_08.txt','star1_5_33.txt','star2_1_61.txt','star3_1_29.txt','star4_1_40.txt']
st0 = star_values(filee[0])
st1 = star_values(filee[1])
st2 = star_values(filee[2])
st3 = star_values(filee[3])
st4 = star_values(filee[4])

#print(st0) # hver inneholder tre arrayelementer med tid i antall dager, bølgelengde og fluks

#

def values(star):
    s_l = len(star)                  # Finding number of data files
    time = np.zeros(s_l).tolist()    # An array to hold all the time values of the stars
    velocity = np.zeros(s_l).tolist()# Same for velocity instead of time
    flux = np.zeros(s_l).tolist()    # Same for flux instead of time
    for i in range(s_l):             # Looping over all the stars
        t,l,f = star_values(star[i]) # Getting measured values of the star
        v = (l-lambda0)*light_speed/lambda0  # convert wavelength to velocity
        time[i], velocity[i], flux[i] = t,v,f # Putting values in right array

    return time,velocity,flux

star_values = values(filee)
time,vel,flux = star_values[0],star_values[1],star_values[2]
#print(time)
#print(np.amax(star_values[1][0]))
#print(np.amin(star_values[1][0]))

"""
Values:
Function for getting values from the stars given in the star argument, and
computing the velocity of the star from the wavelenght.
Takes a list of files (strings) with stored star values (star), the
wavelenght for H_2 molecules (l0) and the speed of light (c) as arguments.
"""
"""
Function for computing the peculiar velocity.
Takes measured velocity (vel) as arguments.
"""
# Skal så finne pekuliærhastighet
# obs_v = pek_v + banehast_stjerne

def peculiar_velocity(vel):
    peculiar_vel = np.zeros(len(vel))       # An array for holding the pec. velocity of the stars
    for n in range(len(vel)): # Looping over the number of stars
        ####calculate peculiar velocity here
        peculiar_vel[n] = np.mean(vel[n]) #(np.amax(vel[n]) + np.amin(vel[n]))/2
    return peculiar_vel

pec_vel = peculiar_velocity(vel)
#print(pec_vel)
#print(time[1])
"""
plotvf
Function for plotting the different velocities and flux of the stars.
Takes time (tid), velocity (hastighet), peculiar_velocity  and flux
(fluks) arrays for all the stars as arguments. That means the arrays are
on the form [star number, measured value]
"""

def plot_v_f(tid,hastighet,peculiar,fluks):

    fig = plt.figure()              # Creating a figure
    #fig.suptitle("Radial velocities of five given stars")
    num_stars = len(tid)            # Finding number of stars
    for i in range(num_stars):      # Looping over the different stars
        vre = hastighet[i]-peculiar[i]
        ax = fig.add_subplot(num_stars,1,i+1)       # Creating a subplot in the figure
        ax.plot(tid[i],vre) #****Insert whatever you want to plot****
        #ax.xaxis.set_ticks(np.arange(np.amin(tid[i]),np.amax(tid[i]+1),1000))
        #ax.yaxis.set_ticks(np.arange(np.amin(vre[i]),np.amax(vre[i]+1),5))
    plt.xlabel("Time, [days]")
    plt.ylabel("Radial velocity, [m/s]")
#    plt.savefig("velocities")
    plt.show()

    fig = plt.figure()
    #fig.suptitle("Flux of five given stars")
    for i in range(num_stars):      # Same as over, but plotting flux instead
        ax = fig.add_subplot(num_stars,1,i+1)
        ax.plot(tid[i],fluks[i]) #****Insert whatever you want to plot****
        #ax.xaxis.set_ticks(np.arange(np.amin(tid[i]),np.amax(tid[i]+1),1000))
    plt.xlabel("Time, [days]")
    plt.ylabel("Relative flux, [m/s]")
#    plt.savefig("flux")
    plt.show()
# ax.xaxis.set_ticks(np.arange(start, end, stepsize))
# for label in cbar.ax.xaxis.get_ticklabels()[::2]:
#    label.set_visible(False)
plot_v_f(time,vel,pec_vel,flux)
### planet for stjerne 3,4,5
### v: 15 - 4200 dager, 15 - 4000 dager, 20 - 4800 # for tre ulike stjerner

#4)
def vr_model(t,t0,P,vr):
    """
    A model function to fit the measured velocity (a normal sin/cos-function).
    Takes the time interval of the star that the function shall model (t) and
    test values for t0 (time the velocity is greatest), period (P) and radial
    velocity (vr) as arguments.
    """
    vr_t = vr*np.cos(2*np.pi/P*(t-t0))
    #calculate the model for all time indices

    return vr_t

## array med mulige verdier for periode,t0 og vr (maks)
t01_prob = np.linspace(2600,3600,20)
t02_prob = np.linspace(1100,1600,20)
t03_prob = np.linspace(1600,2200,20)
t0s =np.array([t01_prob,t02_prob,t03_prob])

vr1_prob = np.linspace(5,10,20)
vr2_prob = np.linspace(12,20,20)
vr3_prob = np.linspace(8,16,20)
vrs =np.array([vr1_prob,vr2_prob,vr3_prob])

P1_prob = np.linspace(4000,4300,20)
P2_prob = np.linspace(4400,5200,20)
P3_prob = np.linspace(5000,5800,20)
ps =np.array([P1_prob,P2_prob,P3_prob])

#test = np.zeros((3,2,5)) # tre underklasser, hver av disse har to underelem med fem elem hver
#test[0][1][3] = 4
#print(np.where(test == np.max(test)))
# husk at vr er max radial hastighet her

def best_delta(t0,P,vr,vel,peculiar,times):
    """
    Method of least squares for finding the best t0, period and radial velocity
    for the vr_model to fit the measured values.
    Takes an interval of test values for t0 (time the velocity is greatest),
    period (P) and radial velocity (vr), and the measured velocity (vel),
    peculiar velocity (peculiar) and time interval of the star that the function shall
    model (times) as arguments.
    """
    #nb! The following code is the simplest, but not the most elegant way to solve this.
    #You may use meshgrid to make a faster and more elegant code,
    #consult the Numpy section of the Numerical Compendium if you wish to learn

    vr_data = vel - peculiar   # Computing the star's radial velocity correcting for peculiar velocity

    len_t,len_p,len_vr = len(t0),len(P),len(vr)     # Finding the lenght of the intervals
    delta_values = np.zeros((len_t,len_p,len_vr))   # Defining a matrix to hold the differece the function computes
    #calculate all elements of  delta_values here # antall arrays: t, antall underarray p, antall verdier vr

    #∆(t , P, v ) = 􏰂sum(vdata(t)−vmodel(t, t , P, v ))^2
    for i in range(len_t):
        for j in range(len_p):
            for k in range(len_vr):
                delta_values[i][j][k] = np.sum((vr_data - vr_model(times,t0[i],P[j],vr[k]))**2)

    #print(delta_values)
    i,j,k = np.where(delta_values == np.min(delta_values)) # Finding the indices for the smallest differece, insert test
    t_i = i[0]; p_j = j[0]; vr_k = k[0] # Turning the list with indices to index numbers

    return t_i,p_j,vr_k, delta_values[t_i,p_j,vr_k] # Returning the indices and minimal value

best_fit_arr = np.zeros((3,4))

for i in range(2,5):
    #print(i)
    best_fit = best_delta(t0s[i-2],ps[i-2],vrs[i-2],vel[i],pec_vel[i],time[i])
    print(best_fit)
    best_fit_arr[4-i,:] = best_fit # in reverse star 5,4,3

# creating value arrays corresponding to plot
best_st3 = list(best_fit_arr[0][:])
best_st2 = list(best_fit_arr[1][:])
best_st1 = list(best_fit_arr[2][:]) # t,p,v values
b_f = np.array([best_st1,best_st2,best_st3])

best_t0_arr = np.array([t01_prob[int(b_f[0,0])],t02_prob[int(b_f[1,0])], t03_prob[int(b_f[2,0])]])
best_P_arr = np.array([P1_prob[int(b_f[0,1])],P2_prob[int(b_f[1,1])], P3_prob[int(b_f[1,2])]])
best_v_arr = np.array([vr1_prob[int(b_f[0,2])],vr2_prob[int(b_f[1,2])],vr3_prob[int(b_f[2,2])]])
print('Beste modell:')
print('t0 fit')
print(best_t0_arr)
print('\n')
print('P fit')
print(best_P_arr)
print('\n')
print('v fit')
print(best_v_arr)

## For å plotte modellen med dataene for planet 3-5
def plot_vr_model(tid,hastighet,peculiar,best_v,best_p,best_t):   # best fit er array
    """
    Function for plotting the radial velocity data against the model for radial
    velocity for a star over time.
    Takes the time interval (tid), measured velocity (hastighet) and peculiar
    velocity for the star and the best found t0 (best_t), radial
    velocity (best_v) and period (best_p) for the vr_model as arguments.
    """
    fig = plt.figure()              # Creating a figure
    #fig.suptitle("Radial velocities of five given stars")
    num_stars = len(tid)            # Finding number of stars
    for i in range(3):      # Looping over the different stars
        vre = hastighet[i+2]-peculiar[i+2]
        ax = fig.add_subplot(3,1,i+1)       # Creating a subplot in the figure
        ax.plot(tid[i+2],vre) #****Insert whatever you want to plot****
        ax.plot(tid[i+2],vr_model(tid[i+2],best_t[i],best_p[i],best_v[i]))
        #ax.xaxis.set_ticks(np.arange(np.amin(tid[i]),np.amax(tid[i]+1),1000))
        #ax.yaxis.set_ticks(np.arange(np.amin(vre[i]),np.amax(vre[i]+1),5))
    plt.xlabel("Time, [days]")
    plt.ylabel("Radial velocity, [m/s]")
    plt.show()

    #up to you how you want to plot it
plot_vr_model(time,vel,pec_vel,best_v_arr,best_P_arr, best_t0_arr)
### Beregne massen til planetene fra den beste modellen
### eventuelt samtidig med anslag fra kun å se på kurve

def mass_planet(m_star,v_star,p,g):
    """
    Function for computing the lower limit mass of an orbiting planet around the
    star. Assume i = 90.

    Takes mass of the star (m_star), best fitting radial velocity (v_star) and
    best fitting period (p) for the star, and the gravitational constant as arguments.
    """
    return m_star**(2/3)*v_star*p**(1/3)/(2*np.pi*g)**(1/3)


def r_dens_planet(v_star,m_star,m_planet,tt):
    v_planet = v_star*m_star/m_planet
    radius_p = ((v_star + v_planet)*(tt[1] - tt[0]))/2
    density = m_planet*solar_mass/(4/3*np.pi*radius_p**3)
    return radius_p, density

# Skal beregne massen til planeten rundt stjernene 3-5
print('\n')
dtosek = 60*60*24

mass_p = np.zeros(3)
for i in range(3):
    mass_p[i] = mass_planet(m_star[i+2]*solar_mass,best_v_arr[i],best_P_arr[i]*dtosek,G)
    print('masse til planet {:} er {:} kg'.format(i+3,mass_p[i])) #index beg 0

# Skal beregne radius og tetthet til de samme planetene om mulig
flux_tt = np.array([168*dtosek,169*dtosek])
print('\n')
radius, density = r_dens_planet(best_v_arr[1],m_star[1]*solar_mass,mass_p[1],flux_tt)
print('radius planet:{:.4e} m, tetthet: {:} kg/m^3'.format(radius,density))

### v: 15 - 4200 dager, 15 - 4000 dager, 20 - 4800 # for tre ulike stjerner
v_eye = np.array([15,15,20])
p_eye = np.array([4200,4000,4800])
print('\n')
mass_p = np.zeros(3)
for i in range(3):
    mass_p[i] = mass_planet(m_star[i+2]*solar_mass,v_eye[i],p_eye[i]*dtosek,G)
    print('øyemål: masse til planet {:} er {:} kg'.format(i+3,mass_p[i])) #index beg 0

# Skal beregne radius og tetthet til de samme planetene om mulig
print('\n')
radius, density = r_dens_planet(v_eye[1],m_star[1]*solar_mass,mass_p[1],flux_tt)
print('øyemål:radius planet:{:.4e} m, tetthet: {:} kg/m^3'.format(radius,density))

"""
Lasse$ python del_1c.py
(10, 19, 12, 81684.400499376497)
(6, 0, 0, 88809.360771559557)
(1, 0, 15, 92260.875724048296)
Beste modell:
t0 fit
[ 3126.31578947  1257.89473684  1631.57894737]


P fit
[ 4300.  4400.  5000.]


v fit
[  8.15789474  12.          14.31578947]


masse til planet 3 er 1.7022454311127498e+27 kg
masse til planet 4 er 2.1766868613447397e+27 kg
masse til planet 5 er 2.8617257131101387e+27 kg


radius planet:2.5253e+09 m, tetthet: 6.417731742506855e+28 kg/m^3


øyemål: masse til planet 3 er 3.1054815302316947e+27 kg
øyemål: masse til planet 4 er 2.6357754411243725e+27 kg
øyemål: masse til planet 5 er 3.943965518670288e+27 kg


øyemål:radius planet:2.6070e+09 m, tetthet: 7.063906105591251e+28 kg/m^3
"""
