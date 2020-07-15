###  Task 2 ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# figure parameters
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

# Want to draw the phase space (P,p)
N = 10000
m1 = 100
m2 = 1
v1_0 = 1
v2_0 = 0

def phase_space(m1,m2,v1_0,v2_0):
    v1 = np.zeros(N); v2 = np.zeros(N)
    p2 = np.zeros(N); P1 = np.zeros(N)
    v1[0] = v1_0; P1[0] = m1*v1[0]

# m1 >= m2
    for i in range(N-1):
        if (i+1)%2 == False:
            v2[i+1] = -v2[i]
            v1[i+1] = v1[i]
            p2[i+1] = m2*v2[i+1]; P1[i+1] =  m1*v1[i+1]
        else:
            v1[i+1] = (m1-m2)/(m1+m2)*v1[i] + 2*m2*v2[i]/(m1+m2)
            v2[i+1] = 2*m1/(m1+m2)*v1[i] - (m1-m2)/(m1+m2)*v2[i]
            p2[i+1] = m2*v2[i+1]; P1[i+1] =  m1*v1[i+1]
        if v1[i+1] <= 0 and v2[i+1]<= 0 and v1[i+1] <= v2[i+1]:
            ending = i+1
            break
    return ending,P1,p2


def plot_regular():
    plt.grid()
    plt.plot(P1[0:ending],p2[0:ending],'-*')
    plt.title('M:m = {:}'.format(m1))
    plt.xlabel('P1 (large mass)')
    plt.ylabel('p2 (small mass)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# With scaling
def plot_scaling():
    plt.grid()
    P1_twiddle = P1/np.sqrt(2*m1); p2_twiddle = p2/np.sqrt(2*m2)
    plt.plot(P1_twiddle[0:ending],p2_twiddle[0:ending],'-*')
    plt.title('M:m = {:}'.format(m1))
    plt.xlabel(r'$\tilde{P1}$ (large mass)')
    plt.ylabel(r'$\tilde{p2}$ (small mass)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

ending,P1, p2 = phase_space(m1,m2,v1_0,v2_0)
plot_regular()
plot_scaling()

# Calculating distances between successive points:
# We get that distances between successive points
# will be 2*theta*(n) # where n >= 2, is the total number of collisions

# Want to find total number of collisions
# Array of masses
m2 = np.array([1,1,1,1])
m1 = np.array([10,100.00,1e4,1e8])
theta = np.arctan(np.sqrt(m2/m1))
print('theta values:',theta)

m = 0
for i in range(len(m1)):
    while m <= np.pi/theta[i]:
        n = m
        m = n + 1
    print('for large mass={:.2f}m, number of collisions, n ={:.0f},n*angle:{:}'.format(m1[i],n,n*theta[i]))
