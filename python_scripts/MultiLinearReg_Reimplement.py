# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:46:21 2022
ReImplement Algo
@author: TSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import dataset & Extract Features
# =============================================================================

dataset = pd.read_csv("02Companies.csv")

X = dataset.iloc[:, [0,2]].values    # X array
y = dataset.iloc[:, 4].values   # y array


# =============================================================================
# Train Test Split 80-20 
# =============================================================================

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

print("b0 / Intercept ", regressor.intercept_)
print("b1 / Slope ", regressor.coef_)

# =============================================================================
# Model Testing
# =============================================================================

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test,y_pred)   # old 83502864.03257737
                                            # new 67220275.37568115
errorUnits = np.sqrt(error)   # old 9137.990152794944
                              # new 8198.797190788484

from sklearn.metrics import r2_score 
score = r2_score(y_test,y_pred) # old 0.9347068473282425
                                # new 0.9474386447268489























