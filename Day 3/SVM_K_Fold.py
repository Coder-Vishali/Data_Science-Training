# -*- coding: utf-8 -*-
"""
SVM K Fold
@author: vsriniva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("04SUV_Ad.csv")

# =============================================================================
# Feature Extraction
# =============================================================================
X = dataset.iloc[:,[2,3]].values     # 2d array matrix
y = dataset.iloc[:,4].values         # 1d array

# =============================================================================
# Train Test Split, test_size=0.25
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# =============================================================================
# Scaling of Values
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()
scObj.fit(X_train)         # Learn the scale / scaling on train data

X_train= scObj.transform(X_train)
X_test = scObj.transform(X_test)

# To know the range of the scale: Default -2 to 2
# np.min(X_train) # -1.9931891594584856
# np.max(X_train) # 2.3315320031817324

# X_train[0,:] # array([ 0.58164944, -0.88670699])
# Perform Inverse transform 
# scObj.inverse_transform(X_train[0,:]) # array([   44., 39000.])

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.svm import SVC
#classifier = SVC(kernel='linear')
#classifier = SVC(kernel='poly', degree = 5)
classifier = SVC(kernel='rbf')
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing, confusion matrix, accuracy score
# =============================================================================
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,
                             cv=10)

print("Mean", accuracies.mean()) # 0.9033333333333333
print("Std", accuracies.std()) # 0.06574360974438671
# More variation

X_trainmore =np.append(X_train, X_train,axis=0)
y_trainmore =np.append(y_train, y_train,axis=0)

accuracies2 = cross_val_score(estimator=classifier,
                             X=X_trainmore,
                             y=y_trainmore,
                             cv=10)
                             
print("Mean", accuracies2.mean()) # 0.9099999999999998
print("Std", accuracies2.std()) # 0.03511884584284246
