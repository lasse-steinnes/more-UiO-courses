
# coding: utf-8

# # Simulating a Storm Cloud in 3D

# ## Introduction

# This notebook simulates a storm cloud, modeling it as a parallel-plate capacitor. It is based in large part on information from a report called "The Physics of Lightning" by Dwyer and Uman (*Dwyer, J. R., & Uman, M. A. (2014). The physics of lightning. Physics Reports, 534(4), 147–241. http://doi.org/10.1016/j.physrep.2013.09.004*. You can also find a short description of the basic concepts at http://hyperphysics.phy-astr.gsu.edu/hbase/electric/lightning.html#c1). The code calculates the electric field a certain distance away from the cloud, at a specified observation position. There are some suggestions for investigation questions and the end, but you are welcome to play around with the parameters in the code, add or subtract different pieces, and see how it behaves for yourself as you decide what you want to use it to investigate.

# Notater: Koden beregner E-felt en viss distanse fra skyen. Kan undersøke hva som skal til for at lynet skal slå ned i en viss høyde, og hva som skjer ved disse høydene ved ulike lag/ulik høyde på sky og ulik ladningstetthet. Eventuelt hvor lett lynet slår ned i ulike typer grunn (Slår lynet lettere ned i USA enn Oslo, Norge?) Gjøre massive forenklinger mhp. resistivitet og hva bedrocken inneholder.
#
# Teori:
# Skyen er en parallell-platekondensator, med øverste delen positiv og nederste negativ (pluss et tynt lag positivt - p-region som gjør at negative ladninger frigis slik at lynet slår ned). Det er kjøligere høyere oppe, slik at ladningene forblir delt. På bakken blir det dannet en positivt ladet grunnskygge.
#
# Dannelse av spenningen kan skje enten ved dannelse av positivt ladet iskrystaller og tåke ved kollisjoner, eller at varme vinder fører positive ladninger fra jordas overflate til toppen av skya, mens kosmisk stråling danner negative partikler som samles i bunnen av skya. Lette iskrystallene og hagl kolliderer i et medium av superkjølet vann (-10 til -20 C).  Topp: 10 km, bunn 6 km over havnivå.
#
# Lynutslag når spenningen er sterk nok til å ionisere luft.
# For å vite hvor lynet slår ned må man anta lineært medium, ellers holder ikke teorien.
#
# De fleste lynnedslag (skyformasjoner?) er 5-10 km i utsstrekning vertikalt, tordenstorm 6-10*2 km horisontalt. Mange typer nedslag, så stor forenkling.
#
# Pockels: magnetisk felt indusert i basalt stein.
#
#
#

# ## Simulation of the Storm Cloud

# First, we import our various libraries: sympy, numpy, and matplotlib

# In[2]:


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Now, we will define some constants: the dimensions of the cloud, altitude of the cloud, charge of the positive and negative part of the cloud, and the observation position (where we are observing the electric field from the cloud

# In[3]:


startx = -2500 #Define where the parts of the clouds should start/stop in the x direction
endx = 2500

startz = -2500 #Define where the cloud should start/stop in the z direction
endz = 2500

negheight = 6000 #set the negative cloud at a height (in the y direction) of 6000m
posheight = 80000 #set the positive cloud at a height of 8000m
belowground = 750 # eventuelt kan jeg endre på dette direkte i utregning

Q = -15 #Q is the total charge on the bottom (negative) part of the cloud
Q2 = 15 #Q2 is the total charge of the top (positive) part of the cloud
Q3 = 1*10**(-1.478)  #Q3 is total charge on the ground due to polarization

k = 9e9 #Coulomb's constant


pos = np.array([0,0,0]) #the observation position (start at 0,0,0)

# ønsker å konstruere array med flere slike verdier, slik at E-feltet kan beregnes utover et grid,
#slik kan man finne verdier der lynet kan slå ned
# Poeng: Du må ha polarisering i bakken, som bidrar til E-feltet.
n = 100 # steps endret fra 100
pos1 = np.zeros(shape =(n,3))
obspos = np.zeros(shape = (n,3))
x_pos = np.linspace(-2500,2500,n)
z_pos = np.linspace(-2500,2500,n)
z_eq = np.linspace(200,585,n)
#x, y = np.meshgrid(x_pos,y_pos,(x_pos**2)/(10**4.5) - (y_pos**2)/(10**4.5) + 300)

y_pos = (x_pos**2)/(10**4.5) - (z_eq**2)/(10**4.5) + 800 # som en sadel

for i in range(len(x_pos)):
    obspos[i] = np.array([x_pos[i],y_pos[i],z_pos[i]]) # husk at y er høyde !!!!
    pos1[i] = np.array([x_pos[i],750,z_pos[i]])

opos= np.transpose(obspos)
#print(min(opos[2,:]),max(opos[2,:]))
#print(obspos)
#print(pos1)


# Now, we will define the chunks we break the clouds (and net charge) into. First, we'll define how many "chunks" we'll break the cloud into in the x and z direction, the size of each of those chunks (based on the overall size of the cloud) and the charge of each of those chunks

# In[4]:


nx = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)
nz = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)

stepx = (endx - startx)/nx #Define the spacing between each chunk in the x direction
stepz = (endz - startz)/nz #Define the spacing between each chunk in the z direction

dQ = Q/(nx*nz) #Charge of each chunk of the the negative part of the cloud: Denne vil få negativ verdi
dQ2 = Q2/(nx*nz) #Charge of each chunk of the positive part of the cloud
dQ3 = Q3/(nx*nz)
print('Ladningstetthet for overflaten er {:.2e} (øvre skylag), {:} (nedre skylag) og {:} (substrat)'.format(dQ,dQ2,dQ3))


# Finally, we will define the e-field variable, initialize it to 0, and calculate the net e-field by iterating over each of these chunks and adding each of their contributions to the net e-field.
#
# (Note that "np.linalg.norm" essentially takes the magnitude of a vector or array, so we use that in calculating the e-field)

# In[ ]:

"""
# For valley
# ønsker å vektorisere finne ut hvor det mest sannsynlig slår ned
E_arr = np.zeros(n)

for itk in range(n): # må gjøre noe med denne, ps: ikke bland iterator og konstant k, lol
    pos = obspos[itk][:]
    efield = 0
    for i in range(0,nx): #iterate over the x dimension of the cloud
        xloc = startx + i*stepx
        for j in range(0,nz): #iterate over the z dimension of the cloud
            zloc = startz + j*stepz

            negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
            posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
            groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
            efield = efield + negfield + posfield + groundfield
    E_arr[itk] = efield
    if efield < -3*10**6:
        print('Lightning!! E-field value:{:}, position: {:} i:{:}'.format(efield,pos,itk))
#print(E_arr)


# In[ ]:


### Duplicate of code, just with variable height, but straight plane
# ønsker å vektorisere finne ut hvor det mest sannsynlig slår ned.
E_arr = np.zeros(n)

for itk in range(n): # må gjøre noe med denne, ps: ikke bland iterator og konstant k, lol
    efield = 0
    pos = pos1[itk][:]
    for i in range(0,nx): #iterate over the x dimension of the cloud
        xloc = startx + i*stepx
        for j in range(0,nz): #iterate over the z dimension of the cloud
            zloc = startz + j*stepz

            negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
            posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
            groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
            efield = efield + negfield + posfield + groundfield
    E_arr[itk] = efield
    if efield < -3*10**6:
        print('Lightning!! E-field value:{:}, position: {:} (i {:})'.format(efield,pos,itk))
print(E_arr)
"""

# In[ ]:


### Duplicate of code, just with point
efield = 0

pos = np.array([0,1500,0])
for i in range(0,nx): #iterate over the x dimension of the cloud
    xloc = startx + i*stepx
    for j in range(0,nz): #iterate over the z dimension of the cloud
        zloc = startz + j*stepz

        negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
        posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
        groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
        efield = efield + negfield + posfield + groundfield

if efield < -3*10**6:
    print('Lightning!! E-field value:{:}'.format(efield))

print("The e-field at the observation position is", efield, "Newtons per coulomb")

"""
# With 100 by 100 chunks, we get an e-field of -1391.788 N/C. From other tests this is about .002% different from the value with 1000 by 1000 chunks, so it seems that 100 chunks in x and z will work just as well as 1000.

# In[5]:


# gjør beregninger for V og I her gitt R = rho*l/A_tverrsnitt og (electrical breakdown of air 3 x 10^6)
# Da må man beregne integralet, dvs sum av E*dl, r = 2.5*10**-2 m /2
# må la y variere mellom 5 og 5995 for å beregne potensialet
efield = 0
v_diff = 0

dl = (5995)/1000
for y in range(0,5995,1000):
    pos = np.array([0,y,0])
    dl = (5995)/1000
    for i in range(0,nx): #iterate over the x dimension of the cloud
        xloc = startx + i*stepx
        for j in range(0,nz): #iterate over the z dimension of the cloud
            zloc = startz + j*stepz

            negfield = k*dQ/(np.linalg.norm(pos-np.array([xloc,negheight,zloc])))**2
            posfield = k*dQ2/(np.linalg.norm(pos-np.array([xloc,posheight,zloc])))**2
            groundfield = -k*dQ3/(np.linalg.norm(pos-np.array([xloc,pos[1]-0.1,zloc])))**2
            efield = efield + negfield + posfield + groundfield
            v_diff = -efield*dl



print("The v_diff", v_diff, "V")

obs_point = 0
# Strømmen som går gjennom er
l = negheight - obs_point
r = (2.5*10**-2)/2
A =  np.pi*r**2
rho = 1.3*10**(-3.2)
R = rho*l/A
I = v_diff/R
print('strøm:',I)

#30,000 amperes (30 kA) kan lynet gi fra seg


# In[ ]:


# dette spres til bakken og man kan beregne hvor langt unna man må stå for å være sikker.
# bruk konduktivitet
# Antar at Js = I/(pi*r**2) (I/A) sprer seg mest ut på overflaten,
# og at J = sigma*E, slik at E = J/sigma (1/sigma = rho)
# En spenning mellom bena (potensialforskjell) er
# int_r ^(r+d) I/sigma*pi*r**2 dr
sp.init_printing(use_unicode=False, wrap_line=False)
r = sp.Symbol('x')
sigma = sp.Symbol('sigma')
I = sp.Symbol('I')
pi = sp.Symbol('pi')
sp.integrate(I/(sigma*pi*r**2),r)


# In[28]:


steps = 1000
rmaks = 1250
r = np.linspace(1,rmaks,steps) # radius
d = 0.5 # meter mellom benene
# kvarts,feltspat vanlig i gneis
# sandstein # kvart og feltspat, men bundet med leire

#gneiss
g_sigma100 = 100e-3
g_sigma20 = 50e-3

#sandstein
sigma100 = 10e-3 # S/m til et spesielt rock  sandstein
sigma20 = 1e-3 # til et annet  # varer ikke lenge, men rekker å varme opp litt

def sigma(r,sigma20,sigma100):   # pga. varme
    if  r < 1:
        s_val = sigma100
    else:
        s_val = sigma20
    return s_val


def evaluate_V(r,d,sigma20,sigma100,steps):
    V = np.zeros(steps)
    for i in range(steps):
        V[i] = I/(np.pi*sigma(r[i],sigma20,sigma100))*(1/r[i] - 1/(r[i]+d))
    return V

def find_stream(V,r,rmax,sigma20,sigma100):
    I = np.zeros(steps)
    for i in range(steps):
        rho = 1/sigma(r[i],sigma20,sigma100)
        R = rho*r[i]/(2*np.pi*rmax*0.001)
        I[i] = V[i]/R
    return I


gneiss = evaluate_V(r,d,g_sigma20,g_sigma100,steps)
sandstein = evaluate_V(r,d,sigma20,sigma100,steps)

#print(gneiss)
#print(sandstein)


# 100 and 200 mA (0.1 to 0.2 amp) are lethal Strømmen dreper
stream_gneiss = find_stream(gneiss,r,rmaks,g_sigma20,g_sigma100)
stream_sandstein = find_stream(sandstein,r,rmaks,sigma20,sigma100)

print(stream_gneiss)
print(stream_sandstein)
# Bra, da har jeg beregnet strømmen, og jeg har fått verdier
# Da gjenstår kun å plotte og skrive artikkelen


# ## Additional questions you might investigate
#
# 1. How likely is it for lightning to strike a particular spot, knowing that the electric field breakdown of air is about 3e6 V/M? How far away from the cloud would you have to be to be safe from lightning?
#     * What if the cloud polarizes the ground, or an object near the ground?
# 2. Actual clouds have a certain thickness. How does this calculation change if the clouds are not thin sheets, but are 3D instead?
# 3. It turns out that in reality, there are multiple layers of + and - charge, each with somewhat different charge densities (see *Marshall, T. C., & Stolzenburg, M. (1998). Estimates of cloud charge densities in thunderstorms. Journal of Geophysical Research, 103(D16), 19769–19775.*) What happens if there are extra layers of positive or negative charge in these clouds?
# 4. What if the cloud is larger or smaller? Higher up or closer to the ground?
#
# *(Note that these are just meant to be suggestions—feel free to investigate any question you find interesting!)*
#
# For more information on clouds and lightning (including approximate numbers for many of the physical characteristics of storm clouds) see *Dwyer, J. R., & Uman, M. A. (2014). The physics of lightning. Physics Reports, 534(4), 147–241. http://doi.org/10.1016/j.physrep.2013.09.004*
"""

Ved hjelp av en enkel parallellplatekondensator-modell for en stormsky, beregner programmet E-feltet fra skya, og spenningen feltet gir. Programmet kan benyttes til å undersøke hva som skal til for at det blir lynnedslag, og hvor langt unna det er trygt å stå fra nedslaget for å overleve.

For at det skal lyne må spenningen overgå 3e6 V/m. Dette skjedde kun om bakken var polarisert, i samsvar med hvordan lyn oppstår i virkeligheten. Uten det ekstra bidraget til E-feltet, og dermed spenningen, blir det ikke lyn. Siden E er proporsjonal med 1/r^2, blir E-feltet sterkere med høyde, slik at sannsynligheten for lyn, øker med høyden. Skyen er ikke uendelig i utstrekning, og har derfor ikke translasjonssymmetri. Et punkt midt under skya får mest bidrag fra hver E-feltet som oppstår fra hver del av skylaget i modellen. Dermed er det lurt å befinne seg i utkanten av skylaget, og helst utenfor, for å unngå lynet fullstendig.

Ifølge programmet blir resistiviteten i luft tilnærmet null idet lynet slår ned. Dette tilsvarer en konduktivitet tilnærmet uendelig, akkurat slik man ville forvente i plasma. Det er kjent at gassen i lufta omdannes til plasma ved ekstremt høye temperaturer, som for eksempel ved dielektrisk sammenbrudd. Temperaturen i lyn kan være opp mot 27 000 celsius, hele fem ganger varmere enn soloverflaten. Da eksisterer gassen i form av frie, ioniserte elementærpartikler, som gir en ekstremt god ledningsevne. En konsekvens er at strøm kan transporteres over store avstander. Resultatet samsvarer dermed med tidligere kunnskap.

Grunnet dårligere konduktivitet ble det antatt at mindre strøm fordeles over landoverflaten i sandstein enn gneiss. Det betyr at mindre strøm bidrar til spenningen som dannes i sedimentet. Imidlertid gir høyere resistivitet en større spenning mellom to punkter i sedimentet, fordi det er større ladningsforskjeller mellom punktene. Anta at en person står på bakken med en avtstand 0.5 meter mellom benene. Som følge av ladningsforskjellene blir strømmen opp til  mellomgulvet for personen større i sandstein på tross av mindre strøm. Dette gjenspeiler at i et materiale med lav konduktivitet, vil strømmen heller finne andre veier å gå, som for eksempel via huden på et menneske, eller enda mer skadelig; gjennom skjelettmusklene og indre organer. Den dødelige radius for sandstein blir dermed noe større enn for gneiss, som har høyere konduktivitet.

Koden kan benyttes videre for å undersøke strømmen i andre materialer, eller eventuelt andre egenskaper ved stormskyer og lyn.
