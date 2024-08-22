# DATA READ - REYNOLDS AND VELOCITY VALUES
import pandas as pd
import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import data

file_name = 'fxReynoldsCFD_ANN.xlsx'
file_path = os.path.join(os.getcwd(), file_name)

ANNDF = pd.read_excel(file_path)

X = ANNDF.iloc[:,[0,2]].values
Y = ANNDF.iloc[:, 1].values.reshape(-1,1)

xtrain, xtest, ytrain,  ytest = train_test_split(X, Y, train_size = 0.85)

scalerx = StandardScaler()
scalery = StandardScaler()

Xtrain = scalerx.fit_transform(xtrain)
Ytrain = scalery.fit_transform(ytrain)



mlp = MLPRegressor(max_iter=5000)

#-----------------------


# parameter grid, KFold cross-validation

kf = KFold(n_splits=10, shuffle=True)

parameters = {
    'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (4, 4, 4), (6, 6, 6), (8, 8, 8), (4, 4, 4, 4), (6, 6, 6, 6), (8, 8, 8, 8)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam']
}

#applying gridsearchcv, train the model

gsearchcv = GridSearchCV(estimator=mlp, param_grid=parameters, cv=kf)
gsearchcv.fit(Xtrain, Ytrain.ravel())

#define best score and parameters based on R²
bestac = gsearchcv.best_score_ * 100  # Convert to percentage
bestparam = gsearchcv.best_params_

print('Best Accuracy: {:.2f} %'.format(bestac))
print('Best Parameters:', bestparam)


#-------------

# Definir os intervalos para Re e Stdp
Re_vals = np.linspace(200, 8000, 100)  # Valores de Reynolds de 1000 a 8000
Stdp_vals = np.linspace(0.3326, 0.4331, 100)  # Valores de Stdp de 0.3326 a 0.4331

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


from pysr import PySRRegressor


# Reajustar o modelo com PySR
model = PySRRegressor(
    niterations=100,  # Começar com menos iterações para evitar sobrecarga
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["cos", "exp", "sin", "inv(x)=1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},  # Definir a função `inv(x)`
    loss="loss(x, y) = (x - y)^2",  # Função de perda personalizada
    population_size=10,  # Tamanho da população reduzido para evitar sobrecarga
    progress=True,  # Mostrar o progresso
    verbosity=1,  # Nível de verbosidade
)

# Gerar previsões com o modelo já treinado para a superfície
predicted = gsearchcv.predict(Xtrain)
predicted = scalery.inverse_transform(predicted.reshape(-1, 1)).flatten()

# Ajustar o modelo PySR aos dados
# Usar os dados de treinamento após o escalonamento inverso
Re_values = xtrain[:, 0]
Stdp_values = xtrain[:, 1]

# Certificar que X_data e predicted tenham o mesmo número de amostras
X_data = np.column_stack((Re_values, Stdp_values))
model.fit(X_data, predicted)

# Mostrar as equações encontradas
print(model)

# Prever novos valores com a melhor equação simbólica encontrada
best_equation = model.predict(X_data)
print("Best symbolic equation prediction:", best_equation)