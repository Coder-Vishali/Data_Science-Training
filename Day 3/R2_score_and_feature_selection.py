"""
R2 score and feature selection
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
XNew = pd.get_dummies(X,columns=["State"])
XDrop = pd.get_dummies(X,columns=["State"],drop_first=True)
X = pd.get_dummies(X,columns=["State"],drop_first=True)
X = X.values     # Dataframe to numpy 2d array
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
error = mean_squared_error(y_test,y_pred)   
errorUnits = np.sqrt(error)  

######################################
# R2 score
######################################
from sklearn.metrics import r2_score 
score = r2_score(y_test,y_pred)

######################################
# Feature Selection
######################################
'''
Index(['RNDSpend', 'Administration', 
       'MarketingSpend', 'State', 'Profit'], dtype='object')
'''
datasetEval = pd.read_csv("02Companies.csv")
datasetEval = pd.get_dummies(data=datasetEval, columns=["State"], drop_first=True)

from scipy.stats import pearsonr
pearsonr(datasetEval['RNDSpend'], datasetEval['Profit'])
pearsonr(datasetEval['MarketingSpend'], datasetEval['Profit'])
pearsonr(datasetEval['Administration'], datasetEval['Profit'])
pearsonr(datasetEval['State_Florida'], datasetEval['Profit'])
pearsonr(datasetEval['State_New York'], datasetEval['Profit'])
