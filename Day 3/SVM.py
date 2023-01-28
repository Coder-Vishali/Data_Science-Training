"""
Predict Discrete values
@author: Vishali
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
# =============================================================================
# Classification Visualization
# =============================================================================
from matplotlib.colors import ListedColormap

X_set, y_set = X_test,y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,1].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,
             classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))
'''
# General
plt.contourf(X1,X2,
             classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))

# Increase the scale
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-2,stop=X_set[:,1].max()+2,step=0.01),
                     np.arange(start=X_set[:,1].min()-2,stop=X_set[:,1].max()+2,step=0.01))

# To get only the boundary
plt.contourf(classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))

# plt.scatter(X_test[:,0],X_test[:,1])
plt.scatter(X_test[y_test==0,0],X_test[y_test==1,1])
plt.show()
'''

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c=ListedColormap(('red','green'))(i), label = j)
plt.title('Classifier')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
