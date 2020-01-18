# Kjøre programmet og beskrive hva det gjør
"""
from random import random  ### importerer noe
# random genererer pseudorandom tall

antfeil = 0; N = 10000         ### Lagrer som int-objekt
x0 = y0 = z0 = 0.0             ### Lagrer som float-objekt
feil1 = feil2 = 0.0            ### Setter to floatvariable lik hverandre
for i in range(N):             ### Feil 2 er duplikat av feil 1
    x = random(); y = random(); z = random() # setter x, y og z til tilfeldig tall
    res1 = (x + y) + z         ### Res 1 og Res 2 skal være like ifølge
    res2 = x + (y + z)         ### aritmetikkens lover
    if res1 != res2:           ## om res 1 ikke er lik res 2
        antfeil += 1           ## Plusser 1 til 0
        x0 = x; y0 = y; z0 = z ## Setter disse til de tilfeldige
        ikkeass1 = res1        ## Setter inn for res1
        ikkeass2 = res2        ## Setter inn for res2
print (100. * antfeil/N)       ## Printer ut antall ganger lovene slår feil for beregning i prosentandel
print (x0, y0, z0, ikkeass1 - ikkeass2) ## Printer ut differansen mellom summene for siste feilberegning
"""

# Kjøreeksempel
"""
16.99
0.028061657929104755 0.7041557662095211 0.5151782207563695 2.220446049250313e-16
"""

# b)
from random import random

antfeil = 0; N = 10000
x0 = y0 = z0 = 0.0
feil1 = feil2 = 0.0

for i in range(N):
    x = random(); y = random(); z = random()
    res1 = x*(y + z)
    res2 = x*y + x*z
    if res1 != res2:
        antfeil += 1
        x0 = x; y0 = y; z0 = z
        ikkeass1 = res1
        ikkeass2 = res2
print (100. * antfeil/N)
print (x0, y0, z0, ikkeass1 - ikkeass2)

## Kjøreeksempel
"""
31.33
0.5410488672168328 0.3287353559468871 0.6302209147031296 1.1102230246251565e-16
"""

"""
Kommentar: Man ser at feilprosenten er mye høyere,
den forsterkes ved multiplikasjon fordi produktet som genereres
er et mye høyere siffer, og avrunding må foretas om sifferet inneholder mange
desimaltall.
"""
