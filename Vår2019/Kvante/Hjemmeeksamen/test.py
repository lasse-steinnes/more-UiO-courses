## test med imaginære tall
import numpy as np
im = np.complex(0,1)
nu = np.exp(np.pi/4*im)
print(nu)
print(np.conj(nu))
print(np.imag(nu))
print(np.real(nu))
