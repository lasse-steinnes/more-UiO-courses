## 1
"""
t = []; v = [];
infile = open('running.txt','r')
for line in infile:
    tnext, vnext = line.strip().split(',')
    t.append(float(tnext))
    v.append(float(vnext))
infile.close()
"""
"""
Eksempel på eulers midtpunktsmetode
x_arr2 = np.zeros(n+1)
x_arr2[0] = 1/10
# print(x_arr)
t_arr2 = np.zeros(n+1)
for i in range(1,n+1):
    x_halvveis = x_arr2[i-1] + (h/2)*g(t_arr[i], x_arr2[i-1])
    x_arr2[i] = x_arr2[i-1] + h*g(t_arr[i-1]+h/2,x_halvveis)
    t_arr[i] = t_arr[i-1] + h
"""
"""
a = 0
s = [0]
integral = 0
for i in range(1,len(t)):        # fordi i er fra 1 opp til n
    h = (t[i] - t[i-1])
    v_mid = (v[i] + v[i-1])/2               # Beregne v for x_mid
    integral = integral + h*v_mid
    s.append(integral)
"""
# Gaussfordeling
"""
n = 100
sigma = 2
mu = 29
x = np.linspace(30,60,n)
h = (60-30)/n

def p(x, sigma,mu):
    res = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
    return res

integral = 0
for i in range(1,n):        # fordi i er fra 1 opp til n
    h = x[i] - x[i-1]
    p_mid = abs((p(x[i],sigma,mu) - p(x[i-1],sigma,mu))/2)
    integral += h*p_mid
print('Sannsynligheten er:{:.2e}'.format(integral))

## 2: OK

## 3. Sannsynlighet for temp over 30: ssyv dager på rad.
##
p7 = integral**7
print('Sannsynlighet for temp over 30 over 7 dager er:{:.2e}'.format(p7))
"""
