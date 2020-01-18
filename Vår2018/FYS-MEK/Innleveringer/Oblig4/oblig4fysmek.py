##### Oblig 4   #####
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

U0 = 150
m = 23
x0 = 2
alfa = 39.48

n = 10000            # tidssteg
T = 10              # tid [sek]
dt = T/n            # ∆t, ett tidssteg
a = np.zeros(n)
v = np.zeros(n)     # Array for fart
x = np.zeros(n)     # Array for x posisjon
t = np.zeros(n)

def F(x,v):
    return -(U0*x/(x0*abs(x)) + alfa*v)

def run_euler(ix,iv):
    x[0] = ix
    v[0] = iv

    for i in range(n-1):
        if abs(x[i]) > x0:
            a[i] = 0
        else:
            a[i] = float(F(x[i],v[i]))/m
        v[i+1] = v[i] + dt*a[i]
        x[i+1] = x[i] + dt*v[i+1]
        t[i+1] = t[i] + dt

    plt.figure(figsize = (10,5))
    ax1 = plt.subplot(211)
    plt.title(r'Posisjon og hastighet til partikkelen ($v_{0}$: %0.1f, $x_{0}$: %0.1f)'%(iv,ix))
    plt.plot(t,v)
    plt.ylabel('Hastighet')
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax1.xaxis.set_major_locator(loc)

    ax2 = plt.subplot(212)
    plt.plot(t,x)
    plt.xlabel('tid')
    plt.ylabel('Posisjon')
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax2.xaxis.set_major_locator(loc)
    plt.show() # avtagget for linjene 58-70.
    return x

# Kjører for v0 = 8, x = -5
run_euler(-5,8)

# Kjører for v0 = 10, x = -5
run_euler(-5,10)

# k, finne max v0 slik at partikkelen blir fanget
# sier at ix fortsatt er -5
# Må ligge mellom 8 og 10, går ut av fella ved 10.

iv = np.linspace(8,10,100)
ix = -5

for i in range(100):
    iv_max = iv[i]
    run_euler(ix,iv[i])
    if x[-1] > 0.2:
        break
print(r'Den største verdien til v0 er {:0.2f}.'.format(iv_max))
