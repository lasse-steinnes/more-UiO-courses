### Programs for midterm exam questions  ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#### Task 1.e. Plotting the phase space

# Define constants
m = 0.5       # kg
g = 9.81    # m/s^2
alpha = np.pi/3# Choosing 45 degree half angle
p_phi = np.sqrt(m*np.sin(alpha)**2) # Choosing the constant angular momenta


################################################################
########## By using cylindrical coordinates ##########################
# Limits
r_min = 0.001 # cannot be negative
r_lim = 2
p_r_lim = 1.2 # can be negative

# Making a grid
r_list = np.linspace(r_min,r_lim,1000)
p_r_list = np.linspace(-p_r_lim,p_r_lim,1000)

r, p_r = np.meshgrid(r_list, p_r_list)

# rho is same in rows, p_rho is same in columns

#
r_dot = p_r/(m*(1+1/np.tan(alpha)**2))
p_r_dot = +p_phi**2/m*2/r**3 \
            - m*g/np.tan(alpha)

scaling = 1
# Use streamplot to get the phase space

# figure parameters
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

plt.streamplot(r*scaling, p_r*scaling, r_dot*scaling, p_r_dot*scaling, density=1.0, linewidth=2, cmap = "autumn")
plt.xlabel(r'$r$ [m]')
plt.ylabel(r'$p_{r}$ [kg m s$^{-1}$]')
plt.title(r'Phase space for the cone ($p_{\theta}$):'+' {:.1f}'.format(p_phi)+r' [kg m$^{2}$ s$^{−1}$]')
#plt.axis('equal')
plt.tight_layout()
plt.savefig('cone_phase_space2.pdf')
plt.show()


### Using spherical coordinates
"""
# Limits
rho_min = 0.001 # cannot be negative
rho_lim = 2
p_rho_lim = 1 # can be negative

# Making a grid
rho_list = np.linspace(rho_min,rho_lim,1000)
p_rho_list = np.linspace(-p_rho_lim,p_rho_lim,1000)

rho, p_rho = np.meshgrid(rho_list, p_rho_list)

# rho is same in rows, p_rho is same in columns

#
rho_dot = p_rho/m
p_rho_dot = p_phi**2/(m*np.sin(alpha)**2)*1/rho**3 \
            - m*g*np.cos(alpha)

#
scaling = 1
# Use streamplot to get the phase space

plt.streamplot(rho*scaling, p_rho*scaling, rho_dot*scaling, p_rho_dot*scaling, density=1.0, linewidth=2, cmap = "autumn")
plt.xlabel(r'$\rho$ [m]')
plt.ylabel(r'$p_{\rho}$ [kg m s$^{-1}$]')
plt.title('Phase space for the cone')
#plt.axis('equal')
plt.tight_layout()
plt.savefig('cone_phase_space.pdf')
plt.show()
"""
###
