""" Oppgaver til kap. 12.2"""

### oppg. 2
## Skal approksimere
## intervall [0,1] f = x**2

##
b = 1
a = 0
n = 2 # (to delintervaller,dvs to midtpunkter)
h = (b-a)/n

integral = 0
for i in range(1,3):        # fordi i er fra 1 opp til n
    x_mid = a + (i-(1/2))*h
    integral += h*(x_mid)**2

print('Oppgave 2')
print("""En approksimasjon til intergralet til f = x**2
på intervallet [0,1] er {}""".format(integral))


## Oppg. 3
##
from math import pi, sin
b = pi/2
a = 0
n = 6 # (to delintervaller,dvs to midtpunkter)
h = (b-a)/n

integral = 0
for i in range(1,7):        # fordi i er fra 1 opp til n
    x_mid = a + (i-(1/2))*h
    integral += h*sin(x_mid)/(1+(x_mid)**2)

print('\nOppgave 3')
print("""En approksimasjon til intergralet til f = sin(x)/(1+x**2)
på intervallet [0,pi/2] er {}""".format(integral))


### Oppg 4 # a
from math import pi, sin,exp, e, sqrt
b = 1
a = 0
n = 10 # (to delintervaller,dvs to midtpunkter)
h = (b-a)/n

integral = 0
for i in range(1,11):        # fordi i er fra 1 opp til n
    x_mid = a + (i-(1/2))*h
    integral += h*exp(x_mid)

print('\nOppgave 4 a)')
print("""En approksimasjon til intergralet til f = e**(x)
på intervallet [0,1] er {}""".format(integral))
print('Dvs. e - 1 = {}'.format(e-1))

## Oppg 4 # b
# Bestem en verdi av h som garanterer at absoluttfeilen er mindre enn 10**(-10)

## Må finne en max f´´(x) på intervallet
import numpy as np
def d2_f(x):
    return np.exp(x)
x_arr = np.linspace(0,1,100) # Hadde ikke trengt dette. e**x er jo økende på hele intervallet. Dvs e**b er størst!
M = np.max(d2_f(x_arr))
b = 1
a = 0
err = 10**(-10)
h_max = sqrt(24*err/((b-a)*M))
print('\nOppgave 4 b)')
print("""Om absolutt feilverdi til f = e**(x) skal være
{} på intervallet [0,1] er h mindre enn = {:.4e}""".format(err,h_max))



## Oppgave 6
## På papir
