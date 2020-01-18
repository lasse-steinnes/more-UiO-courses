## til oppgave 2
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

v_h = np.array([1.1,6.2,11.7,17.3,23,28.6])
b_arr = np.array([6,42,74,108,142,175])
b_new = np.linspace(0,180,100) # en serie punkter
n = 1# tilpass til et n-te gradspolynom

coefs = poly.polyfit(b_arr, v_h, n)
print('stigningstall',coefs)
ffit = poly.polyval(b_new, coefs)

plt.plot(b_new, ffit,'-')
plt.plot(b_arr,v_h,'o')
plt.xlabel('mT')
plt.ylabel('mV')
plt.show()

"""
# B-felt
t = 1*10**(-2)
a = 1.75*10**(-2) #
mu0 = 4*np.pi*10**(-7) #Ns**2/C**2
Js = 9.0109*10**5
h_arr = np.array([0,2,4,6,8,10])*10**(-2)
h = np.linspace(0,10,100)*10**(-2)
B_measured = np.array([282,70.5,17,7,4,2.6])
B_measured = B_measured*(10**-3) -1.1*10**(-3)

def B_x(h,t,a):
    ans = mu0/2*Js*((h+t)/np.sqrt((h+t)**2+a**2) - h/np.sqrt(h**2 + a**2))
    return ans

Bx = B_x(h,t,a)
print(Bx)

##
plt.plot(h, Bx,'-')
plt.plot(h_arr,B_measured,'o')
plt.xlabel('h')
plt.ylabel('B')
plt.show()
##

coefs = poly.polyfit(h_arr, B_measured, 5)
print('stigningstall',coefs)
ffit = poly.polyval(h, coefs)

plt.plot(h, ffit,'-')
plt.show()

### Skal gjøre det samme, men for spole med js
N = 244
l = 275*10**(-3)
I = 5
Js = (N*I)/l
print('Js',Js)
a = 4*10**(-2)

h_arr2 = np.array([0,1,2,4,6,8,10])*10**(-2)
h2 = np.linspace(0,10,100)*10**(-2)
B2_measured = np.array([2.7,2,1.4,1.0,0.5,0.3,0.2])
B2_measured = B2_measured*(10**-3)

B2 = B_x(h2,l,a)

print(Bx)
print(B2_measured,h_arr2)
plt.plot(h2, B2,'-')
plt.plot(h_arr2,B2_measured,'o')
plt.xlabel('h')
plt.ylabel('B')
plt.show()
"""
