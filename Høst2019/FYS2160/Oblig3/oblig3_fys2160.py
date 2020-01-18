### Obligatory assignment 3 ###
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


#Task 1 and 2
# without change of reference s
ei = np.array([-0.1,-0.05,0,0.05,0.1]) # eNirgy levels

# With change of reference
ei2 = ei + 0.1
k =  8.617333262145*10**(-5) # boltzmann [eV/K]
T = 300              # [K]

def Z(en):
        return np.sum(np.exp(-en/(k*T)))

def prob(en):
    return np.sum(np.exp(-en/(k*T)))

######
for i in range(len(ei)):
#    print('Partition'+ str(i) +':' + str(Z(ei[i])))
    print('Probability'+str(i)+':'+str(1/Z(ei)*prob(ei[i])))

print('Total probabilities:'+str(1/Z(ei)*prob(ei)))
print('Total Partition:' + str(Z(ei)))

print('\n------------------------ \n With change of reference\n--------------------')
for i in range(len(ei2)):
#    print('Partition'+ str(i) +':' + str(Z(ei2[i])))
    print('Probability'+str(i)+':'+str(1/Z(ei2)*prob(ei2[i])))

print('Total probabilities:'+str(1/Z(ei2)*prob(ei2)))
print('Total Partition:' + str(Z(ei2)))


### Exercise 3 ###
k = 1.38*10**(-23); NA = 6.022*10**23; MNi = 0.028; mN = MNi/NA
T = np.array([300,600])

### calculating speeds
v_avg = np.sqrt(8*k*T/(np.pi*mN))
v_rms = np.sqrt(3*k*T/mN)
v_p = np.sqrt(2*k*T/mN)

print('v_avg:{:} v_rms:{:} v_p:{:} m/s for T = [300,600]'.format(v_avg,v_rms,v_p))

######################

def gasspeed3D(v,T, m = mN):
    return 4*np.pi*v**2*((m/(2*np.pi*k*T))**(3/2))*np.exp(-m*v**2/(2*k*T))

# set v
N = 10000
v = np.linspace(0,2500, N)

# Cumulative probabilities using cumsum

plt.figure(12, figsize = (8,6))
ax1 = plt.subplot(121)
ax1.plot(v,gasspeed3D(v,T = 300), label = '300 K')
ax1.plot(v,gasspeed3D(v,T = 600), label = '600 K')
ax1.legend()
ax1.set_xlabel('Speed [m/s]')
ax1.set_ylabel('D(v)')

delta_v = 1/4
ax2 = plt.subplot(122)
ax2.plot(v,np.cumsum(gasspeed3D(v,T = 300)*delta_v), label = '300 K')
ax2.plot(v,np.cumsum(gasspeed3D(v,T = 600)*delta_v), label = '600 K')
ax2.set_xlabel('Speed [m/s]')
ax2.set_ylabel('Cumulative probability')
plt.show()

## Task 3
### Task 4 - evaluating the integral numerically.
# Find probability of moving faster than 11km/s
def gasspeed3D(v):
    return 4*np.pi*v**2*((m/(2*np.pi*k*T))**(3/2))*np.exp(-m*v**2/(2*k*T))

T = 300
m = MNi
prob1 = integrate.quad(gasspeed3D,0,300)
print('Task 3 prob:{:}'.format(prob1))

upper_limit = 11*1000 # m/s
T = 1000
v = np.linspace(0,upper_limit,N)

### Task 5
MH = 0.002 ; mH = MH/NA; MHe = 0.004; mHe = MHe/NA
masses = np.array([mN,mH,mHe])
### for Hydrogen

print('\n-----------------\nProbability on earth\n---------------\n')

for i in range(len(masses)):
    m = float(masses[i])
    prob = integrate.quad(gasspeed3D,upper_limit,upper_limit*30)
    print('Probability of particle:'+str(prob))

# A bit higher for Hydrogen

### On the moon
upper_limit = 2.4*10**3 # m/s

print('\n-----------------\nProbability on moon\n---------------\n')

for i in range(len(masses)):
    m = float(masses[i])
    prob = integrate.quad(gasspeed3D,upper_limit,upper_limit*30)
    print('Probability of particle:'+str(prob))
# Gas particle is moving too fast. So the atmosphere cannot hold it
################


'''
Output:



'''
