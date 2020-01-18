##### Matriseoperasjoner ####

# implementasjon av algoritmen for gauss eliminasjon

# Steg 1) Forward
# Anta at man har matrise (array) M
# Om n er antall rader/kolonner i matrisa med a-er

for m in range(1,n):   # loop for matrisa
    for i in range(m+1,n):  # loop for rad
        for j in range(m+1,n): # loop for kolonne
            C[i,j] = M[i+1,j]/M[i,j]
            Mhat[i,j] = M[i,j] - M[i-1,j]*C[i,j]


# x list
# Steg 2) Backward løsning for x
sum_ = [0]*n
x[n] = b[n]/a[n][n]

for n in reversed(range(1,n-1)):
    for m in range(1,n):
        sum_[n] = a[n][n+1]*x[n+1]
    x[n] = (b[n] - sum_[n])/a[n][n]
