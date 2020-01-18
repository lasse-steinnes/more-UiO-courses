## trying reverse ##

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

"""
l = np.linspace(0,10,11)

for i in reversed(range(0,11)):
    print(i)
import autograd.numpy as np
import autograd as autograd

def f(z):
    return z**2

gradz = autograd.grad(f,0)
z = np.linspace(0,4,10)
print(gradz(2))

list_ = np.array([0,1,4,3])
print(np.min(list_))

x = np.zeros((2,10)) + 10
a,b = x

print(x)

def probabilities(f_z):
    print('sum',np.sum(np.exp(f_z)))
    a,b = f_z
    return np.exp(f_z)/np.sum([np.exp(a),np.exp(b)], axis = 0)


print(probabilities(x))
"""
filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

target = (df[['default payment next month']].copy()).to_numpy()
data = (df.drop(columns =['default payment next month', 'ID']).copy()).to_numpy()

scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print(data)
