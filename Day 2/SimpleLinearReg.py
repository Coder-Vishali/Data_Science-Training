"""
SimpleLinear Regression =>Predict Continuos values
@author: Vishali
"""
# =============================================================================
# import Libraries & data
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("01HR_Data.csv")

# =============================================================================
# X & Y Feature Extraction
# =============================================================================
X = dataset.iloc[:,[0]].values       # 2d matrix
y = dataset.iloc[:, 1].values       # 1d array

# =============================================================================
# Train - Test Split , test_size=1/3
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)        # Machine Learned 
print("Slope / b1 ", regressor.coef_)
print("Intercept / b0 ", regressor.intercept_)

# =============================================================================
# Model Testing
# =============================================================================
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test,y_pred)
errorUnits = np.sqrt(error)

# =============================================================================
# Add-On Visualization
# =============================================================================
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,c="Red")
plt.show()
