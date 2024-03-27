import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np

df = pd.read_excel('ANN_TRAIN_TEST_RAMON_T40_AIR_WATER.xlsx')
X = df.iloc[:,1:7].values
Y = df.iloc[:, [12,13]].values


#Parameter control
trainsize = 0.75
neurons = 30
activation_function = 'relu'
batchsize = 32
epochs = 1000
hidden_layers_quantity = 8

#normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size = trainsize, random_state=0)

#Building the ANN MODEL


ann = tf.keras.models.Sequential() #Initializethe ann model

#creatin a sequence of hidden layers (adding according to the range)
for _ in range(hidden_layers_quantity):
    ann.add(tf.keras.layers.Dense(units=neurons, activation=activation_function, kernel_regularizer=regularizers.l2(0.01)))
    ann.add(tf.keras.layers.Dropout(0.2)) #DROPUT LAYER

ann.add(tf.keras.layers.Dense(units = 2)) # OUTPUT LAYER


#Compiling the ann, it adds the stochastic gradient descent method to control losses
ann.compile(optimizer = 'adam', 
            loss = 'mean_squared_error') #COMPILER (TRY DIFFERENT PARAMETERS)

#TRAIN THE ANN MODEL
ann.fit(Xtrain, Ytrain, 
        batch_size = batchsize, 
        epochs = epochs)

Qpred = ann.predict(Xtest)
Xtrain = scaler.inverse_transform(Xtrain)
Xtest = scaler.inverse_transform(Xtest)
X = scaler.inverse_transform(X)

graf_exp_water = go.Scatter(x = X[:, 5], y = Y[:,0], mode = 'markers', name = 'EXPERIMENTAL_WATER',marker = dict(size = 10,
                                                                        color = 'purple',
                                                                        symbol = 'pentagon'))

graf_exp_air = go.Scatter(x = Xtest[:, 5], y = Ytest[:,1], mode = 'markers', name = 'EXPERIMENTAL_AIR',marker = dict(size = 10,
                                                                        color = 'purple',
                                                                        symbol = 'pentagon'))

Qtest_water = go.Scatter(x = Xtest[:, 5], y = Ytest[:, 0], mode = 'markers', name = 'Qtest_WATER',marker = dict(size = 11,
                                                                        color = 'green',
                                                                        symbol = 'x',
                                                                        line = {'width':2}))

Qtest_air = go.Scatter(x = Xtest[:, 5], y = Ytest[:, 1], mode = 'markers', name = 'Qtest_AIR',marker = dict(size = 11,
                                                                        color = 'green',
                                                                        symbol = 'x',
                                                                        line = {'width':2}))

Qpred_water = go.Scatter(x = Xtest[:, 5], y = Qpred[:, 0], mode = 'markers', name = 'Qpred_WATER',marker = dict(size = 7,
                                                                        color = 'red',
                                                                        symbol = 'circle'))

Qpred_air = go.Scatter(x = Xtest[:, 5], y = Qpred[:, 1], mode = 'markers', name = 'Qpred_AIR',marker = dict(size = 7,
                                                                        color = 'red',
                                                                        symbol = 'circle'))



data_water = [graf_exp_water,Qtest_water, Qpred_water]
#data_air = [graf_Qpredair, graf_Qtestair]

layout_water = go.Layout(title = 'ANN_TRAIN_TEST_RAMON_T40_AIR_WATER', 
                   xaxis = dict(title = 'Vazão massica do ar (m_ar)'),
                   yaxis = dict(title = 'Taxa de troca térmica Q_water '))

layout_air = go.Layout(title = 'ANN_TRAIN_TEST_RAMON_T40_AIR_WATER', 
                   xaxis = dict(title = 'Vazão massica do ar (m_ar)'),
                   yaxis = dict(title = 'Taxa de troca térmica Q_air '))

fig_water = go.Figure(data_water, layout_water)

pyo.plot(fig_water)