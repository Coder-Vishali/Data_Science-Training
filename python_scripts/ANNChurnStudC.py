# -*- coding: utf-8 -*-
"""
ANN using Keras Sequential API => Predict Customer Churn
Discrete value - Classification
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
# NEURAL NETWORK IMPLEMENTATION
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense
# =============================================================================
# Initialization of Neural Network
# =============================================================================
classifier = Sequential()

# =============================================================================
# Adding both Input & Hidden Layers using Dense
# First Input Layer adding with 11 Neurons
# Output Layer Predicting Exit column values 0/1,
# binary classification, single neuron in output Layer

## Second Hidden Layer adding Neurons ?????
"""
Minimum Number of Neurons in Hidden Layer == NN Underfits

Number of Neurons in Hidden Layer to start with = 
(NumberOfNeuronsInInputLayer+NumberOfNeuronsInOutputLayer)/2
= (11 + 1) = 6


Maximum Number of Neurons in Hidden Layer == NN OverFits
MaxNumOfNeurons == 5 times the minimum 


Number of Hidden Layers,

if NumOfInputFeatures < 5, start with minimum 1 HL
if NumOfInputFeatures > 5, start with minimum 2 HL
"""
# =============================================================================
## Add InputLayer with 11 Neurons and 1st Hidden Layer with 6 Neurons

classifier.add(Dense(input_dim=11, units=24,kernel_initializer='uniform',activation='relu'))

## Add 2nd Hidden Layer with 6 Neurons

classifier.add(Dense(units=24,kernel_initializer='uniform',activation='relu'))

## Add output Layer , single Neuron, predict 0/1

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

## NN summary

classifier.summary()

classifier.get_layer('dense_3').get_weights()

# =============================================================================
# Compiler Settings for ANN
# Optimizer is the that would do the task of optimal weight finding for your NN
# Algorithm used by Optimizer for Weight Updation is Stochastic Gradient descent
# =============================================================================
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# =============================================================================
# Neural Network Learning, Model Learning
#batch_size = 10          weight update happens after every 10 rows batch   ## mini batch GD
#batch_size = 1            weight update happens after every 1 rows    ##  SGD
#batch_size = 8000 (train dataset size)  weight update happens after entire set of rows   ## batch GD

# =============================================================================
history=classifier.fit(X_train,y_train, batch_size=25,epochs=100)

# =============================================================================
# Model Testing
# =============================================================================
y_pred = classifier.predict(X_test)

y_pred = y_pred>0.5
# =============================================================================
# Accuracy Measure
# =============================================================================

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)



















