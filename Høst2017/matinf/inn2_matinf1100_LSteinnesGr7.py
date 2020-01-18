### Innlevering 2 matinf ###

## Oppgave 1 c
t = []; v = [];
infile = open('running.txt','r')
for line in infile:
    tnext, vnext = line.strip().split(',')
    t.append(float(tnext))
    v.append(float(vnext))
infile.close()


## plot og beregning for numerisk derivasjon v(t)/t
n = len(t)
# Viktig poeng: Ulik steglengde, oppdater h

a = [0]
for i in range(0,n-1):
    h = (t[i+1] - t[i]) #
    a.append((v[i+1]-v[i])/h) # indeksen tilsvarer antall steg

# print(len(a),len(t))
## print(a)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(t,a, label = 'Akselerasjon')
#plt.title('Akselerasjon som en funksjon av tid')
plt.xlabel('Tid')
plt.ylabel('Akselerasjon')
#plt.legend()
plt.savefig('numdiv', format = 'png')
plt.show()


## plot og beregning for numerisk integrasjon s(t)/t


 # (to delintervaller,dvs to midtpunkter)

a = 0
s = [0]
integral = 0
for i in range(1,len(t)):        # fordi i er fra 1 opp til n
    h = (t[i] - t[i-1])
    v_mid = (v[i] + v[i-1])/2               # Beregne v for x_mid
    integral = integral + h*v_mid
    s.append(integral)

plt.figure()
plt.plot(t,s, label = 'Strekning som en funksjon av tid')
#plt.title('Strekning som en funksjon av tid')
plt.xlabel('Tid')
plt.ylabel('Strekning')
plt.savefig('numint', format = 'png')
plt.show()


### Oppgave 2 ###

## Deloppgave a)
##  x = e**t/(9+e**t)

## Deloppgave b)
import numpy as np


def f(t):                     ## Eksakt løsning f
    return np.exp(t)/(9+np.exp(t))

def g(t,x):                     ##  Numerisk løsning g
    return x*(1-x)

a = 0; b = 10;
n = 5; h = (b-a)/float(n)

x_arr = np.zeros(n+1)
x_arr[0] = 1/10
t_arr = np.zeros(n+1)

for i in range(1,n+1):
    x_arr[i] = x_arr[i-1] + h*g(t_arr[i], x_arr[i-1])
    t_arr[i] = t_arr[i-1] + h

## Deloppgave c) Tolker oppgaven som at man skal benytte samme antall steg

## Eulers midtpunktsmetode

x_arr2 = np.zeros(n+1)
x_arr2[0] = 1/10
# print(x_arr)
t_arr2 = np.zeros(n+1)
for i in range(1,n+1):
    x_halvveis = x_arr2[i-1] + (h/2)*g(t_arr[i], x_arr2[i-1])
    x_arr2[i] = x_arr2[i-1] + h*g(t_arr[i-1]+h/2,x_halvveis)
    t_arr[i] = t_arr[i-1] + h

t_arr2 = np.linspace(0,10,1000)

plt.figure()
plt.plot(t_arr,x_arr, label = r'X(t): Eulers metode$_{n=5}$')
plt.plot(t_arr2,f(t_arr2), label = r'X(t): $\frac{e^{t}}{9+e^{t}}$ (Analytisk)')
plt.plot(t_arr,x_arr2, label = r'X(t): Eulers midtpunktsmetode$_{n=5}$')
plt.legend()
plt.xlabel('t')
plt.ylabel('x(t)')
#plt.title(r'Løsninger av $x\prime=x(1-x), x(0)=\frac{1}{10}$')
plt.savefig('numdifferensial', format = 'png')
plt.show()
