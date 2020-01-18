### 2. ny oppgave
# Beregne binomialkoeffisienten

# Binomialkoeffisienten
# a)
# Algoritme/strategi:
# Skal skrive ut produktet av noe gjentatte ganger
# kan laste ned modulen product fra numpy
# eventuelt bruke en for loop, som jeg er kjent med fra tidligere

def Binomial(n,i):
    """Beregner binomialkoeffisienten, oppgis n, i"""
    product_ = float(1)
    for j in range(1,(n-i)+1,1):
        onepart_ = float((i + j)/j)
        product_ *= float(onepart_)
    return product_

print(Binomial(5000, 4))   # Må bruke flyttall fordi man kan lagre mye
                           # større tall på denne måten

print(Binomial(int(1e5),60))

print(Binomial(int(1e3),500))

## observerer at når n blir stor, så blir binomialkoeffisienten mindre.
    # fordi n er med i nevneren.
# Burde skrive inn en testfunksjon her, siden det står et program

# b) Det er mulig å få overflow dersom i + j er stor nok,
# siden binomialkoeffisienten beregnes ut fra dette.

# c)
# Metoden i b kan benyttes når i er veldig stor for da vil i strykes bort.
### Metoden i c kan benyttes når det er stor differanse mellom
# n og i, for da vil delen i nevneren bli nær n i verdi. n blir forkortet
# Metoden i b kan benyttes når i er veldig stor for da vil i strykes bort.
