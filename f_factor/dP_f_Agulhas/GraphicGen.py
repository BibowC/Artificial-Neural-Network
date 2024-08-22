from f_factor.ANN_Model_MLP import gsearchcv

import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo



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

# Calcular a superfície usando a fórmula fornecida
f_formula = (4.3059 / np.log(Re_grid + 3.1918)) + ((0.30649 / Stdp_grid) - 0.80937)

# Criar o gráfico 3D para os resultados da ANN com contornos
surface_ann = go.Surface(
    x=Re_grid, y=Stdp_grid, z=f_pred_grid, 
    colorscale='Viridis',
    name='Superfície ANN',
    colorbar=dict(title='ANN f', len=0.5, x=0.9),
    contours={
        "z": {
            "show": True,
            "start": np.min(f_pred_grid),
            "end": np.max(f_pred_grid),
            "size": (np.max(f_pred_grid) - np.min(f_pred_grid)) / 20,  # Ajustar o espaçamento dos contornos
            "color": "black",
        }
    }
)

# Criar o gráfico 3D para a fórmula fornecida com contornos
surface_formula = go.Surface(
    x=Re_grid, y=Stdp_grid, z=f_formula, 
    colorscale='Plasma',
    name='Superfície Fórmula',
    opacity=0.7,
    colorbar=dict(title='Fórmula f', len=0.5, x=1.05),
    contours={
        "z": {
            "show": True,
            "start": np.min(f_formula),
            "end": np.max(f_formula),
            "size": (np.max(f_formula) - np.min(f_formula)) / 20,
            "color": "black",
        }
    }
)

# Scatter plot dos valores originais
scatter_cor = go.Scatter3d(
    x=Recor1,  # Valores de Reynolds
    y=stpkyan,  # Valores de Stdp
    z=cor2(Recor1),  # Valores de 'f'
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Valores Originais'
)


# Scatter plot dos valores originais
scatter_data = go.Scatter3d(
    x=ANNDF.iloc[:, 0].values,  # Valores de Reynolds
    y=ANNDF.iloc[:, 2].values,  # Valores de Stdp
    z=ANNDF.iloc[:, 1].values,  # Valores de 'f'
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Valores Originais'
)

# Definir o layout do gráfico
layout = go.Layout(
    title='Superfície de Resultados da ANN e da Fórmula com Valores Originais',
    scene=dict(
        xaxis=dict(title='Reynolds Number (Re)', backgroundcolor='white', gridcolor='lightgrey'),
        yaxis=dict(title='Stdp', backgroundcolor='white', gridcolor='lightgrey'),
        zaxis=dict(title='f', backgroundcolor='white', gridcolor='lightgrey'),
        bgcolor='white'
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=True
)

# Criar a figura com ambas as superfícies e o scatter plot
fig = go.Figure(data=[surface_ann, scatter_cor], layout=layout)

# Mostrar o gráfico
pyo.plot(fig)
