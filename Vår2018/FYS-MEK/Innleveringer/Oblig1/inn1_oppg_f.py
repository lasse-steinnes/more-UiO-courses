# Oppgave f
# 100 meter på t sekunder
t1 = np.argwhere(x > 100)
x100 = t[t1]
print('tida ved x=100 er {:.3e}'.format(x100[0][0]))


# Dvs en snitthastighet på
v_ = 100/6.79
print(v_) #
