###### Oppgave 1.
## a)

# x(n+2) + bx(n+1) + cXn = 0 kan skrives om
# Vi har da
"""
x0 = 1
x1 = 2
n = 100

x_list = []
x_list.append(x0)
x_list.append(x1)


for i in range(n+1):
    x = 2*x_list[i+1] + 2*x_list[i]
    x_list.append(x)

x_list2to100 = x_list[2:101]

print("------------------")
print(" n     Xn")
print("------------------")
for i in range(n-1):
    print("%3g  %10.4e" % (i+2, x_list2to100[i]))
print("-------------------")
# print(x_list[2:100])
"""
## b)

from math import sqrt

x0 = 1
x1 = 1-sqrt(3)
n = 100

x_list = []
x_list.append(x0)
x_list.append(x1)


for i in range(n+1):
    x = 2*x_list[i+1] + 2*x_list[i]
    x_list.append(x)

x_list2to100 = x_list[2:101]
"""
print("------------------")
print(" n     Xn")
print("------------------")
for i in range(n-1):
    print("%3g  %10.4e" % (i+2, x_list2to100[i]))
print("-------------------")
"""
##c)
# Se beregning

## d) Kan kjøre en testfunksjon her
## Vise det grafisk med en tabell

x_exactlist = []
e_list = []

for i in range(n+1):
    xn_exact = (1 - sqrt(3))**i
    x_exactlist.append(xn_exact)
    difference_ = abs(float(x_exactlist[i]) - float(x_list[i]))
    e_list.append(difference_)

x_exactlist2to100 = x_exactlist[2:101]
e_list2to100 = e_list[2:101]

#print(x_exactlist2to100) Test
# print(e_list2to100)     Test

print("------------------------------------------------------")
print(" n          Xnum            Xexact            error")
print("------------------------------------------------------")
for i in range(n-1):
    print("%3g  %15.4e  %15.4e  %15.4e" % (i+2, x_list2to100[i],\
    x_exactlist2to100[i], e_list2to100[i]))
print("-------------------------------------------------------")

## Bestemme avrundingsenheten(round off unit)
# Se notat i foredrag om dette
