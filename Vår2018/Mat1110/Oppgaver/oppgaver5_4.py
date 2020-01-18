#### Iterasjon av funksjoner ###
import numpy as np
import matplotlib.pyplot as plt
# 1 Beregn elementer i følgen

n = 1000
x = np.zeros(n)
y = np.zeros(n)
n_arr = np.linspace(0,n,n)

for i in range(1,n): # indeksen begynner i 0
    x[i] = 0.6*x[i-1] - 0.6*y[i-1] + 0.2
    y[i] = 0.6*x[i-1] + 0.6*y[i-1] + 1

plt.plot(n_arr,x, label = r'$x_{n}$')
plt.plot(n_arr,y, label = r'$y_{n}$')
plt.axis([0,400,-2,2])
plt.legend()
plt.show()


# Oppgave 2
# lager som en klasse
class iterasjon():
    def __init__(self,a,b,c,d,e,f,n):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.n = n
    def iterate(self,x,y):
        a = self.a; b = self.b;c = self.c; d = self.d;
        e = self.e; f = self.f; n = self.n

        for i in range(1,n): # indeksen begynner i 0
            x[i] = a*x[i-1] + b*y[i-1] + c
            y[i] = d*x[i-1] + e*y[i-1] + f

        self.x = x
        self.y = y
        return self.x,self.y

    def plot(self):
        n_arr = np.linspace(0,self.n,self.n)
        plt.plot(n_arr,self.x, label = r'$x_{n}$')
        plt.plot(n_arr,self.y, label = r'$y_{n}$')
        plt.legend()
        plt.show()

# 1

n = 400
x = np.zeros(n)
y = np.zeros(n)

it_instanse = iterasjon(0.6,-0.6,0.2,0.6,0.6,1,n)
iterated = it_instanse.iterate(x,y)
it_instanse.plot()

# 2
x = np.zeros(n); y = np.zeros(n)
x[0] = 20; y[0] = 2000
it_instanse = iterasjon(0.9,0.01,-10,-1.01,1,300,n)
iterated = it_instanse.iterate(x,y)
it_instanse.plot()

# 4

class newit(iterasjon):
    def iterate(self,x,y):
        a = self.a; b = self.b;c = self.c; d = self.d;
        e = self.e; f = self.f; n = self.n
        for i in range(1,n):
            x[i] = a*x[i-1] + b*y[i-1] + c
            if e*y[i-1] < x[i-1]:
                y[i] = d*x[i-1] + e*y[i-1] + f
            else:
                y[i] =x[i-1]
        self.x = x
        self.y = y
        return self.x,self.y
n = 400
x = np.zeros(n); y = np.zeros(n)
x[0] = 8; y[0] = 12
ny_instanse = newit(1.01/2,1.01/2,0,0,1.1,0,n)
iterert = ny_instanse.iterate(x,y)
ny_instanse.plot()


n = 200
x = np.zeros(n); y = np.zeros(n)
x[0] = 12; y[0] = 8
ny_instanse = newit(1.01/2,1.01/2,0,0,1.1,0,n)
iterert = ny_instanse.iterate(x,y)
ny_instanse.plot()


# Oppgave 5
# del 1, 2
n = 50
x = np.zeros(n)
y = np.zeros(n)
x[0] = 0.3 #0.1
y[0] = 0.9 #0.8

n_arr = np.linspace(0,n,n)

for i in range(1,n): # indeksen begynner i 0
    x[i] = 2.2*x[i-1]*(1-x[i-1]) + 0.01*y[i-1]*x[i-1]
    y[i] = 3.1*y[i-1]*(1-y[i-1])- 0.02*y[i-1]*x[i-1]

plt.plot(n_arr,x, label = r'$x_{n}$')
plt.plot(n_arr,y, label = r'$y_{n}$')
plt.legend()
plt.show()
