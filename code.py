# DATA READ - REYNOLDS AND VELOCITY VALUES
import pandas as pd
import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import data

file_name = 'fxReynoldsV3.xlsx'
file_path = os.path.join(os.getcwd(), file_name)

ANNDF = pd.read_excel(file_path)

X = ANNDF.iloc[:,[1,3]].values
Y = ANNDF.iloc[:, 2].values.reshape(-1,1)

xtrain, xtest, ytrain,  ytest = train_test_split(X, Y, train_size = 0.95) #30% para teste. e shuffle!

scalerx = StandardScaler()
scalery = StandardScaler()

Xtrain = scalerx.fit_transform(xtrain)
Ytrain = scalery.fit_transform(ytrain)



mlp = MLPRegressor(max_iter=5000)

#-----------------------


# parameter grid, KFold cross-validation

kf = KFold(n_splits=10, shuffle=True)

parameters = {
    'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (8, 16), (16, 16), (16,32), (32,32)], #, (4, 4, 4), (6, 6, 6), (8, 8, 8), (4, 4, 4, 4), (6, 6, 6, 6), (8, 8, 8, 8)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam']
}

#applying gridsearchcv, train the model

gsearchcv = GridSearchCV(estimator=mlp, param_grid=parameters, cv=kf, verbose = 2)
gsearchcv.fit(Xtrain, Ytrain.ravel())

#define best score and parameters based on R²
bestac = gsearchcv.best_score_ * 100  # Convert to percentage
bestparam = gsearchcv.best_params_

print('Best Accuracy: {:.2f} %'.format(bestac))
print('Best Parameters:', bestparam)


#-------------

# Definir os intervalos para Re e Stdp
Re_vals = np.linspace(200, 8000, 50)  # Valores de Reynolds de 1000 a 8000
Stdp_vals = np.linspace(0.3326, 0.4331, 50)  # Valores de Stdp de 0.3326 a 0.4331

# Criar uma grade de Re e Stdp
Re_grid, Stdp_grid = np.meshgrid(Re_vals, Stdp_vals)
grid_points = np.c_[Re_grid.ravel(), Stdp_grid.ravel()]

# Prever os valores de 'f' usando a ANN nos pontos da grade
f_pred_scaled = gsearchcv.predict(scalerx.transform(grid_points))
f_pred = scalery.inverse_transform(f_pred_scaled.reshape(-1, 1))
f_pred_grid = f_pred.reshape(Re_grid.shape)

#---------------------------


input_data = np.array([[1843, 0.4331]])

# Transformar os dados de entrada usando o scaler treinado
input_data_scaled = scalerx.transform(input_data)

# Fazer a previsão com o modelo treinado
predicted_scaled = gsearchcv.predict(input_data_scaled)

# Inverter a transformação do output (scaling)
predicted = scalery.inverse_transform(predicted_scaled.reshape(-1, 1))

Xtest_scaled = scalerx.transform(xtest)

ypred = gsearchcv.predict(Xtest_scaled).reshape(-1,1)

ypred_or = scalery.inverse_transform(ypred)

# Mostrar o resultado
print('Predicted value for [2000, 0.24]:', predicted[0][0])
