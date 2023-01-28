# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:10:42 2022
Hyper Parameter Tunning in automated fashion
@author: TSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Data Import & Feature Extraction
# =============================================================================
dataset = pd.read_csv("Churn Modelling-Bank Customers.csv")

X = dataset.iloc[:, 3:13]          # X dataframe
y = dataset.iloc[:,13].values      # y array

# =============================================================================
# Encode Categorical values using get_dummies
# =============================================================================

X = pd.get_dummies(X,drop_first=True)

X = X.values    # convert it to array

# =============================================================================
# Train Test Split 
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# =============================================================================
# Scaling of Values
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()

X_train = scObj.fit_transform(X_train)
X_test = scObj.transform(X_test)


# =============================================================================
# Hyper Parameter Tunning
# =============================================================================
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier  # type conversion of kerasObj to sklearnObj
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer, h1_size, h2_size):
    classifier=Sequential()
    classifier.add(Dense(input_dim=11, units=h1_size,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=h2_size,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Type cast keras to sklearn

classifier = KerasClassifier(build_fn=build_classifier)    

parameters = {"h1_size":[6,12,24,48] ,
              "h2_size":[6,12,24,48],
              "optimizer":['adam','rmsprop'],
              "batch_size":[10,25,50],
              "epochs":[100,150,500]}

gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)


gridSearch = gridSearch.fit(X_train,y_train)

best_parameters = gridSearch.best_params_
best_accuracy = gridSearch.best_score_







