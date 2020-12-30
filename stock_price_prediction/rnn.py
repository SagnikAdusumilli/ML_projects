# Recurrent Neural Network

"""
Created on Tue Dec 29 16:30:33 2020

@author: Sagnik_laptop
"""
# Train LSTM model on 5 years of stock data (2012-2016) 
# will try to predict the trend on Jan 2017

# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# We only have one feature to predict on (Open price)
train_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())
plt.plot_date(dataset_train.Date, dataset_train.Open, xdate=True)
plt.show()

# Feature Scaling
# RNN's work better with normialized data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)

#Specify data structure for RNN 60 time steps for 1 output
# model will look at 60 prev stock prices in the 
# past to predict the future price
# we can expirement with time steps to balance overfitting and generalization

x_train = [] #will contain 60 prev stock price
y_train = [] #will contain the next day's stock price
for i in range(60, len(train_set_scaled)):
    # get the 60 previous stock prices before ith day
    x_train.append(train_set_scaled[i-60 : i, 0])
    y_train.append(train_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
# RNN takes 3d array of shape (batch_size, time steps, input_dim)
x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], 1))


# Part 2 - Building the RNN

# importing libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Intitialising the RNN
rnn_regressor = Sequential() # sequence of layers

#Add layers and Dropout regularization

#1st layer
# number of lstm neurons
# resturn_sequences = true when you add more lstm layers
# input_shape: shape of the input (only specify time steps and indicators)
rnn_regressor.add(LSTM(units = 50, 
                       return_sequences = True, 
                       input_shape = (x_train.shape[1], 1)))
# add dropout regularization
rnn_regressor.add(Dropout(0.2))

#adding more lstm layers
rnn_regressor.add(LSTM(units = 50, return_sequences = True))
rnn_regressor.add(Dropout(0.2))

rnn_regressor.add(LSTM(units = 50, return_sequences = True))
rnn_regressor.add(Dropout(0.2))

rnn_regressor.add(LSTM(units = 50))
rnn_regressor.add(Dropout(0.2))

#adding output layer
rnn_regressor.add(Dense(units = 1))

#compiling the rnn
# RMS-prop is usually good for RNN (not in this case though)
# mean squared error is used for regression 
rnn_regressor.compile(optimizer='adam', loss='mean_squared_error')

# train the rnn
rnn_regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)
rnn_regressor.save('rnn_regressor.h5')


# Part 3 - Making the predictions and visualising the results
# load the actual stock price
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

# get predicted predicted stock price for jan 2017
# to get 20 predicted stock prices, we'll need 60 prev stock prices for
# each of those days
dataset_total = pd.concat((dataset_train['Open'], test_set['Open']), axis=0)

# to start predection day of the test_set, we need get the last 60 days
# from the test set
inputs = dataset_total[(len(dataset_train) - 60):].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i-60: i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (len(x_test), x_test.shape[1], 1))

predicted_stock_price = rnn_regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google stock price')
plt.plot(predicted_stock_price, 
         color ='blue', label='Predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

# Conclusion:
# our model cannot react very quickly to sudden non leanear change
# but our model can somewat predict trends if the changes are smooth

#possible TODOs:
# add more indicator vaiable 
# (eg stock price of other companies closely related to google)
# add more LSTM layer
# add more neurons to the layer
# get more training data 10 years worth
# incrase time steps to 120 days 