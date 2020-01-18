### Oblig2mat1110 ###
# 1c)
from math import pi

def riemann_f(k,n):
    f = lambda x,y: 1/(x**4 + 2*(x**2)*(y**2) + y**4 +1)

    sum_f = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            dx = - k + 2*k*i/n
            dy = - k + 2*k*j/n
            sum_f += f(dx,dy)*(2*k/n)**2
    return sum_f

print('Kjøreeksempel')
print('Tilnærmet verdi: {:.3f}'.format(riemann_f(19,22)))
print('Eksaktverdi: {:.3f}'.format((pi**2)/2))

"""
Kjøreeksempel
Tilnærmet verdi: 4.909
Eksaktverdi: 4.935
"""

# 1d)
# Skal finne en verdi av k og en verdi av n slik at
# summen nærmer seg integral med mindre enn 1/10 feil.
n = 0
k = 0

for k in range(20):
    for n in range(200):
        riemann = riemann_f(k,n)
        n_last = n
        k_last = k
        if abs((pi**2)/2 - riemann_f(k,n)) < 1/10:
            break

print('Tilnærmet verdi {:.3f}'.format(riemann))
print('n:{:0.3f}'.format(n))
print('k:{:0.3f}'.format(k))

"""
Tilnærmet verdi 4.851
n:22.000
k:19.000
"""
