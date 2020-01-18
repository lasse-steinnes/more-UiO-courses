## Pogramming needed for chapter 2##
# Task 2.8
import numpy as np

def omega(N,q):
    num = np.math.factorial(q+N-1)
    denum = (np.math.factorial(q)*np.math.factorial(N-1))
    return num/denum

qA = np.linspace(0,20,21); qB =  np.flip(qA.copy())
N = 10

omega_tot = 0
for i in range(len(qA)):
    omega_tot += omega(N,qA[i])*omega(N,qB[i])

print('Omega total: {:3e}'.format(omega_tot))

allA = omega(N,20)
zeroB = omega(N,0)
proballA = allA*zeroB/omega_tot

print('all A: {:3e}, Prob all A: {:3e}'.format(allA,proballA))

halfA = omega(N,10)
halfB = omega(N,10)
prob = halfA*halfB/omega_tot
print('halfA: {:3e}, Prob halfA: {:3e}'.format(halfA,prob))

# Task 2.30: Entropy
def omegaL(N,q):
    return (np.exp(1)*q/N)**N
#
N = 10**23
q = 2*N

# a) Entropy, assuming long time, all microstates allowed.
