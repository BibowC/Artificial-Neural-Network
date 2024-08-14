# DATA READ - REYNOLDS AND VELOCITY VALUES
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklear.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#import data

ANNDF = pd.read_excel('fxReynoldsV3.xlsx')

X = ANNDF.iloc[:,[0,2]].values
Y = ANNDF.iloc[:, 1].values.reshape(-1,1)

scalerx = StandardScaler()
scalery = StandardScaler()

X_scaled = scalerx.fit_transform(X)
Y_scaled = scalery.fit_transform(Y)
