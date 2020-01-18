from sklearn import svm
import xgboost as xgb
from sklearn. metrics import r2_score, mean_squared_error
from sklearn import linear_model
import numpy as np
"""
def linear(X,y,regtype):

    -----------------------
    Linear regression method
    ------------------------
    Parameters:
    regtype: Either Ridge, LinearRegression,Lasso

    clf = linear_model.regtype(alpha = 0.1)
    fit = clf.fit(X,y)
    weights = clf.coeff_

X = np.array([[0,0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])
linear(X,y,'Lasso')
"""
#lam_path = [10**(-i) for i in range(7)] + [2*10**(-i) for i in range(1,7)] + [4*10**(-i) for i in range(1,7)] + [6*10**(-i) for i in range(1,7)] +[8*10**(-i) for i in range(1,7)]
#print(lam_path)
#lam = [None, lam_path, lam_path]
