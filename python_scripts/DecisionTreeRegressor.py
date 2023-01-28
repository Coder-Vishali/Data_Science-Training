# -*- coding: utf-8 -*-
"""
DecisionTree Regressor - Regression Tree - Predicting Continuos Values
@author: TSE
"""

# =============================================================================
# Import Libraries
# =============================================================================

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
# Fitting DT Regressor
# min_samples_split
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

regressor.fit(X_train,y_train)

# =============================================================================
# Model Testing 
# =============================================================================
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error_DT = mean_squared_error(y_test,y_pred)


# =============================================================================
# Tree plotting
# =============================================================================

from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file="treePC.dot", feature_names=["Average Salary"])












