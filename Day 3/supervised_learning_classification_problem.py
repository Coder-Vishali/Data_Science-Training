# -*- coding: utf-8 -*-
"""
Supervised learning - Classification
@author: Vishali
"""
# =============================================================================
# Import Libraries and dataset
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("Churn_Modelling_Bank_Customers.csv")

# =============================================================================
# Feature Extraction
# =============================================================================
X = dataset.iloc[:,3:13]    # 2d array matrix
y = dataset.iloc[:,13].values         # 1d array

# =============================================================================
# Dummy variable creation
# =============================================================================
X = pd.get_dummies(X,columns=["Geography"],drop_first=True)
X = pd.get_dummies(X,columns=["Gender"],drop_first=True)
X = X.values     # Dataframe to numpy 2d array

# =============================================================================
# Train Test Split, 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# =============================================================================
# Scaling of Values
# =============================================================================
scObj = StandardScaler()

scObj.fit(X_train)         # Learn the scale / scaling on train data

X_train= scObj.transform(X_train)
X_test = scObj.transform(X_test)

# =============================================================================
# Model Implementation - DecisionTree Classifier
# =============================================================================
classifier = DecisionTreeClassifier( max_depth=2,random_state=0)
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing - DecisionTree Classifier
# =============================================================================
y_pred = classifier.predict(X_test)
cm_DT = confusion_matrix(y_test,y_pred)
acc_DT = accuracy_score(y_test,y_pred)

# =============================================================================
# Model Implementation - Logistic Regression Classifier
# =============================================================================
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing - Logistic Regression Classifier
# =============================================================================
y_pred = classifier.predict(X_test)
cm_LR = confusion_matrix(y_test,y_pred)
acc_LR = accuracy_score(y_test,y_pred)

# =============================================================================
# Model Implementation - Naive Bayes
# =============================================================================
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing - Naive Bayes
# =============================================================================
y_pred = classifier.predict(X_test)
cm_NB = confusion_matrix(y_test,y_pred)
acc_NB = accuracy_score(y_test,y_pred)

# =============================================================================
# Model Implementation - Random Forest
# =============================================================================
classifier = RandomForestClassifier(n_estimators=10,
                                    max_depth=2,
                                    random_state=0)
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing - Random Forest
# =============================================================================
y_pred = classifier.predict(X_test)
cm_RF = confusion_matrix(y_test,y_pred)
acc_RF = accuracy_score(y_test,y_pred)

# =============================================================================
# Model Implementation - SVM
# =============================================================================
classifier = SVC(kernel='rbf')
classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing - SVM
# =============================================================================
y_pred = classifier.predict(X_test)
cm_SVM = confusion_matrix(y_test,y_pred)
acc_SVM = accuracy_score(y_test,y_pred)


# =============================================================================
# K Fold validation
# =============================================================================
accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,
                             cv=10)
print("Mean", accuracies.mean()) 
print("Std", accuracies.std()) 