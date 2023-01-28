# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:11:09 2022
Hyper Parameter Tuning with automated parametre
@author: Vishali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Data Import & Feature Extraction
# =============================================================================
dataset = pd.read_csv("Churn_Modelling_Bank_Customers.csv")

X = dataset.iloc[:, 3:13]          # X dataframe
y = dataset.iloc[:,13].values      # y array

# =============================================================================
# Encode Categorical values using get_dummies
# =============================================================================

X = pd.get_dummies(X,drop_first=True)
# without column names, it will consider all string columns

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
# Hyper parameter Tuning
# =============================================================================
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier # Type conversion of kerasOb to sklearn Obj
from keras.models import Sequential
from keras.layers import Dense

# for more paramters - randomized search cv is better
# Apart from that it is same
def build_classifier(optimizer, h1_size, h2_size):
    classifier = Sequential()
    ## Add InputLayer with 11 Neurons and 1st Hidden Layer with 6 Neurons
    classifier.add(Dense(input_dim=11, units=h1_size,kernel_initializer='uniform',activation='relu'))
    ## Add 2nd Hidden Layer with 6 Neurons
    classifier.add(Dense(units=h2_size,kernel_initializer='uniform',activation='relu'))
    ## Add output Layer , single Neuron, predict 0/1
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    ## NN summary
    classifier.summary()
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Type cast keras to sklearn

classifier = KerasClassifier(build_fn=build_classifier)

parameters={"h1_size":[6,12,24,48],
            "h2_size":[6,12,24,48],
            "optimizer":['adam','rmsprop'],
            "batch_size":[10,20,50],
            "epochs":[100,150,500]}

gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy',cv=10)
gridSearch = gridSearch.fit(X_train, y_train)

RandomizedSearchCV(estimator=classifier, parameters, scoring='accuracy',cv=10)
best_parameters = gridSearch.best_params_
best_accuracy = gridSearch.best_score_
