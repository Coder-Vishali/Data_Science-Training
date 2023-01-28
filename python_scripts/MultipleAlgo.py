# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:22:28 2022

@author: TSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("NewPC.csv")

# =============================================================================
#  Feature Extraction
# =============================================================================
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values

# =============================================================================
# Train Test Split 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Model Implementation
# Fitting Random Forest Regressor
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators=120, random_state=0)
regressorRF.fit(X_train,y_train)

# =============================================================================
# Model Implementation
# Fitting Linear Reg
# =============================================================================
from sklearn.linear_model import LinearRegression
regressorLR = LinearRegression()

regressorLR.fit(X_train,y_train)


# =============================================================================
# Future Prediction
# =============================================================================
UserInput = np.array([[4391]])

predLR = regressorLR.predict(UserInput)
predRF = regressorRF.predict(UserInput)

y_pred = predLR+predRF/2









