## Beregning og plotting av variasjon i de ulike kreftene
def Fc(t):
    return  fc*np.exp(-(t/tc)**2)
def Fv(v):
    return fv*v
def D(t,v):
    return 0.5*A*(1-0.25*np.exp(-(t/tc)**2))\
        *rho*cd*(v**2)

# Beregning av kreftene
F = [F]*len(t1)
Fct = Fc(t1[0:i_max])
Fvt = Fv(v1[0:i_max])
Dt = D(t1[0:i_max],v1[0:i_max])
t_intervall = t1[0:i_max]

# plot
plt.plot(t_intervall, F[0:i_max], label = 'Drivkraft')
plt.plot(t_intervall, Fct, label = 'Initiell drivkraft')
plt.plot(t_intervall,Fvt, label = 'Fysiologisk begrensning')
plt.plot(t_intervall,Dt, label = 'Luftmotstand')
plt.xlabel('Tid [s]')
plt.ylabel('Kraft [$N= kg*m*s^{-2}$]')
plt.legend(loc = 'center right')
plt.ylabel('Kraft [$N= kg*m*s^{-2}$]')
plt.show()
