"""
LSTM for APPLE

@author: Vishali
"""
import numpy as np
import pandas as pd

df=pd.read_csv('AAPL.csv')

df1=df['close']

import matplotlib.pyplot as plt
plt.plot(df1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)  #
test_size=len(df1)-training_size

train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix

def create_dataset(dataset, time_step):
	dataX, dataY = [], []
                   
	for i in range(len(dataset)-time_step-1):            
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
                                                          
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model=Sequential()                                  #timeStep, [close]
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

"""
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,
          batch_size=64,verbose=1)
"""
# =============================================================================
# Save Model Using Keras
# =============================================================================
#model.save("LSTMApple.h5")

# =============================================================================
# Use the Saved Model Using Keras
# =============================================================================
model= keras.models.load_model('LSTMApple.h5')
# =============================================================================

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transform back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
Trainerror = math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
TestError = math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
look_back=100

# shift test predictions for plotting

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[training_size+look_back:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1),c="yellow")
plt.plot(testPredictPlot)
plt.show()