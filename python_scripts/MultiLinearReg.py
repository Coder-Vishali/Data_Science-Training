# -*- coding: utf-8 -*-
"""
Multiple Linear Regression

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import dataset & Extract Features
# =============================================================================

dataset = pd.read_csv("02Companies.csv")

X = dataset.iloc[:, :-1]    # X dataframe
y = dataset.iloc[:, 4].values   # y array

# =============================================================================
# Convert Stats Column to Dummy variable
# =============================================================================

X = pd.get_dummies(data=X,columns=['State'],drop_first=True)

X = X.values   # convert DF to array

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
errorUnits = np.sqrt(error)   # old 9137.990152794944

from sklearn.metrics import r2_score 
score = r2_score(y_test,y_pred) # old 0.9347068473282425

# =============================================================================
# Feature Evaluation
# ['RNDSpend', 'Administration', 'MarketingSpend', 'Profit',
#       'State_Florida', 'State_New York']
# =============================================================================
datasetEval  = pd.read_csv("02Companies.csv")

datasetEval = pd.get_dummies(data=datasetEval,columns=['State'],drop_first=True)


from scipy.stats.stats import pearsonr

pearsonr(datasetEval['RNDSpend'],datasetEval['Profit']) # 0.9729004656594831, 3.5003222436906035e-32)
pearsonr(datasetEval['Administration'],datasetEval['Profit']) #(0.20071656826872136, 0.16221739470358285)
pearsonr(datasetEval['MarketingSpend'],datasetEval['Profit']) # (0.7477657217414766, 4.381073182030992e-10)
pearsonr(datasetEval['State_Florida'],datasetEval['Profit']) # (0.11624426298842244, 0.4214479133054575)
pearsonr(datasetEval['State_New York'],datasetEval['Profit']) # (0.03136760015130278, 0.8287963474107514)























