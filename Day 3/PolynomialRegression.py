"""
Polynomial Regression
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Extract Features
# =============================================================================

dataset = pd.read_csv('Pressure.csv')

X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train,y_test = train_test_split(X,y, test_size = 1/3,
                                                     random_state=0)

# Hyper paramter tuning - here it is polynominal degree is the hyper parameter.
# what degree is best we need to find it out. Start with the minimum degree
# Start with minimum and increase gradually
# =============================================================================
# Add Polynomial Degree
# =============================================================================
from sklearn.preprocessing import PolynomialFeatures
poly_Obj = PolynomialFeatures(degree=5  ,  include_bias=True)
Xtrain_poly = poly_Obj.fit_transform(X_train)
Xtest_poly = poly_Obj.fit_transform(X_test)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain_poly,y_train)

# =============================================================================
# Model Testing
# =============================================================================
y_pred = regressor.predict(Xtest_poly)

from sklearn.metrics import mean_squared_error
# error_2 = mean_squared_error(y_test,y_pred)
# error_5 = mean_squared_error(y_test,y_pred)
# error_10 = mean_squared_error(y_test,y_pred)
#error_15 = mean_squared_error(y_test,y_pred)
error_20 = mean_squared_error(y_test,y_pred)
# =============================================================================
## for higher resolution and continuos curve we increament the tempreture by 0.1
# =============================================================================

X_grid = np.arange(min(X_test),max(X_test),0.1) 
X_grid = X_grid.reshape((len(X_grid), 1))   

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, regressor.predict(poly_Obj.fit_transform(X_grid)),color = 'blue')
plt.title("Polynomial Regression ")
plt.xlabel("Temp")
plt.ylabel("Pressure")
plt.show()
