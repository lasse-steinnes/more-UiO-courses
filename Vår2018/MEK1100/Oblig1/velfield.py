# oppgave 4b) Del 1
# funksjonen beregner hastigheter og lagrer de som array
import numpy as np

def velfield(n):
    x = np.linspace(-0.5*np.pi,0.5*np.pi,n)
    [X,Y] = np.meshgrid(x,x)
    Vx = np.cos(X)*np.sin(Y)
    Vy = -np.sin(X)*np.cos(Y)
    return X,Y,Vx, Vy

"""
Eksempelkall
x,y,vx, vy = velfield(15)
"""
