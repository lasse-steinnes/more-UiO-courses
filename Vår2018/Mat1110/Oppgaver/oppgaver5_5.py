## oppgaver til 5.5 konvergens mot et fikspunkt ##
import numpy as np
import matplotlib.pyplot as plt

# Lage program
def f(x,y):
    return [0.5*np.sin(x+y),0.5*np.cos(x-y)]


# Skal beregne u_n+1 = F(u_n)
def iterasjon(f,u0,N):  # u0 er en arrayvektor
    u = np.zeros((N,2))
    u[0,:] = u0
    n = np.linspace(0,N,100,dtype = 'int')
    for i in range(1,N):
        u[i,:] = f(u[i-1,0],u[i-1,1])

    f, ax1 = plt.subplots()
    ax1.scatter(u[:,0],u[:,1])
    for i,val in enumerate(n):
        ax1.annotate(val,(u[i,0],u[i,1]))
    plt.show()

    return u

# b) Kjøre for u0 = [1,-1] og plotte
u0 = [1,-1]
u = iterasjon(f,u0,100)

# c) Tilfeldig tall mellom -2.5 og 2.5
def iterasjon2(f,u0,N):  # u0 er en arrayvektor
    u = np.zeros((N,2))
    u[0,:] = u0
    n = np.linspace(0,N,100,dtype = 'int')
    for i in range(1,N):
        u[i,:] = f(u[i-1,0],u[i-1,1])
    return u


r = np.random.uniform(-2.5,2.5,size = (6,2))
print(r)

for i in range(len(r)):
    u = iterasjon2(f,r[i,:],100)
    plt.plot(u[:,0],u[:,1],'.-')
plt.show()
"""
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
for j,ax in enumerate(axes.flatten()):
    ax.scatter(u[:,0],u[:,1])
plt.show()
""" # en mulig løsning en senere gang for å plotte subplots i en for løkke.
