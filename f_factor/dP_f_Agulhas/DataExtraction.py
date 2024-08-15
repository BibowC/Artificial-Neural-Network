import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo 
import CoolProp.CoolProp as CP
import tensorflow as tf
from tensorflow.keras import regularizers
import plotly.io
from sklearn.utils import shuffle

#FORMULAS DEF (VEL, REYNOLDS, F)

def Reynolds2(mfr, dh, acs, mu):
    Re = (mfr * dh) / (acs * mu)
    return Re 

def Velocity(mfr, rho, acs):
    Vel = (mfr) / (rho * acs)
    return Vel

def ffactor(dp, dh, L, rho, V):
    f = (dp * dh) / (2*L*rho*(np.power(V,2)))
    return f

dfV3 = pd.read_excel('Dimensions.xlsx', sheet_name='V3')

Vf_cs4V3 = dfV3.iloc[0,1]
Vf_cs6V3 = dfV3.iloc[0,2]
Vf_cs8V3 = dfV3.iloc[0,3]
Vf_cs10V3 = dfV3.iloc[0,4]
Vf_cs12V3 = dfV3.iloc[0,5]
Vf_cs15V3 = dfV3.iloc[0,6]
Afs_cs4V3 = dfV3.iloc[1,1]
Afs_cs6V3 = dfV3.iloc[1,2]
Afs_cs8V3 = dfV3.iloc[1,3]
Afs_cs10V3 = dfV3.iloc[1,4]
Afs_cs12V3 = dfV3.iloc[1,5]
Afs_cs15V3 = dfV3.iloc[1,6]
Acs_cs4V3 = dfV3.iloc[2,1]
Acs_cs6V3 = dfV3.iloc[2,2]
Acs_cs8V3 = dfV3.iloc[2,3]
Acs_cs10V3 = dfV3.iloc[2,4]
Acs_cs12V3 = dfV3.iloc[2,5]
Acs_cs15V3 = dfV3.iloc[2,6]
Dh_cs4V3 = dfV3.iloc[3,1]
Dh_cs6V3 = dfV3.iloc[3,2]
Dh_cs8V3 = dfV3.iloc[3,3]
Dh_cs10V3 = dfV3.iloc[3,4]
Dh_cs12V3 = dfV3.iloc[3,5]
Dh_cs15V3 = dfV3.iloc[3,6]

Lf = 120/1000

# DEFINING REYNOLDS AND VELOCITY
T = 300
mu = CP.PropsSI('V', 'T', T, 'P', 101325, 'Water')
rho = CP.PropsSI('D', 'T', 300, 'P', 101325, 'Water')

Cs4 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs4')
mfr_cs4_V3 = Cs4.iloc[:,9].values
dP_cs4_V3 = Cs4.iloc[:,7].values * 100000
dh_cs4_V3 = [Dh_cs4V3] * len(dP_cs4_V3)
Re_Cs4_V3 = Reynolds2(mfr_cs4_V3, dh_cs4_V3, Acs_cs4V3, mu)
Vel_Cs4_V3 = Velocity(mfr_cs4_V3, rho, Acs_cs4V3)
f_Cs4_V3 = ffactor(dP_cs4_V3, dh_cs4_V3, Lf, rho, Vel_Cs4_V3)

Cs6 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs6')
mfr_cs6_V3 = Cs6.iloc[:,9].values
dP_cs6_V3 = Cs6.iloc[:,7].values * 100000
dh_cs6_V3 = [Dh_cs6V3] * len(dP_cs6_V3)
Re_Cs6_V3 = Reynolds2(mfr_cs6_V3, dh_cs6_V3, Acs_cs6V3, mu)
Vel_Cs6_V3 = Velocity(mfr_cs6_V3, rho, Acs_cs6V3)
f_Cs6_V3 = ffactor(dP_cs6_V3, dh_cs6_V3, Lf, rho, Vel_Cs6_V3)

Cs8 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs8')
mfr_cs8_V3 = Cs8.iloc[:,9].values
dP_cs8_V3 = Cs8.iloc[:,7].values * 100000
dh_cs8_V3 = [Dh_cs8V3] * len(dP_cs8_V3)
Re_Cs8_V3 = Reynolds2(mfr_cs8_V3, dh_cs8_V3, Acs_cs8V3, mu)
Vel_Cs8_V3 = Velocity(mfr_cs8_V3, rho, Acs_cs8V3)
f_Cs8_V3 = ffactor(dP_cs8_V3, dh_cs8_V3, Lf, rho, Vel_Cs8_V3)

Cs10 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs10')
mfr_cs10_V3 = Cs10.iloc[:,9].values
dP_cs10_V3 = Cs10.iloc[:,7].values * 100000
dh_cs10_V3 = [Dh_cs10V3] * len(dP_cs10_V3)
Re_Cs10_V3 = Reynolds2(mfr_cs10_V3, dh_cs10_V3, Acs_cs10V3, mu)
Vel_Cs10_V3 = Velocity(mfr_cs10_V3, rho, Acs_cs10V3)
f_Cs10_V3 = ffactor(dP_cs10_V3, dh_cs10_V3, Lf, rho, Vel_Cs10_V3)

Cs12 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs12')
mfr_cs12_V3 = Cs12.iloc[:,9].values
dP_cs12_V3 = Cs12.iloc[:,7].values * 100000
dh_cs12_V3 = [Dh_cs12V3] * len(dP_cs12_V3)
Re_Cs12_V3 = Reynolds2(mfr_cs12_V3, dh_cs12_V3, Acs_cs12V3, mu)
Vel_Cs12_V3 = Velocity(mfr_cs12_V3, rho, Acs_cs12V3)
f_Cs12_V3 = ffactor(dP_cs12_V3, dh_cs12_V3, Lf, rho, Vel_Cs12_V3)

Cs15 = pd.read_excel('V3inv.xlsx', sheet_name= 'Cs15')
mfr_cs15_V3 = Cs15.iloc[:,9].values
dP_cs15_V3 = Cs15.iloc[:,7].values * 100000
dh_cs15_V3 = [Dh_cs15V3] * len(dP_cs15_V3)
Re_Cs15_V3 = Reynolds2(mfr_cs15_V3, dh_cs15_V3, Acs_cs15V3, mu)
Vel_Cs15_V3 = Velocity(mfr_cs15_V3, rho, Acs_cs15V3)
f_Cs15_V3 = ffactor(dP_cs15_V3, dh_cs15_V3, Lf, rho, Vel_Cs15_V3)

def Traces(x, y, name, color, symbol): #symbol = circle, square, diamond, cross, triangle-up, pentaon, hexagon, star
    trace = go.Scatter(
        x=x,
        y=y,
        name=name,
        mode='markers',
        marker=dict(
            size=15,
            color=color,
            symbol=symbol,  # Forma do marcador
            line=dict(
                width=2,
                color='Black'  # Cor da borda do marcador
            )
        )
    )
    return trace



def curves(trace, title, ytitle):

    fig = go.Figure(data = trace)

    fig.update_layout(
    font=dict(
    size=15),
    title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis={
        'title': 'Reynolds',
        'showline': True,
        'showgrid': False,
        'zeroline': True,
        'zerolinecolor': 'black',
        'linecolor': 'black',
        'ticks': 'outside',
        'mirror': True
    },
    yaxis={
        'title': ytitle,
        'showline': True,
        'showgrid': False,
        'zeroline': True,
        'zerolinecolor': 'black',
        'linecolor': 'black',
        'ticks': 'outside',
        'mirror': True
    },
    plot_bgcolor='white',
    hovermode='closest',
    width=800,  # Largura do gráfico
    height=600  # Altura do gráfico
)
    
    pyo.plot(fig)

    #a = Traces(Re_cs4_V3, abs_cs4V3, 'dp_cs4V3', 'red', 'circle')
b = Traces(Re_cs6_V3, abs_cs6V3, 'dp_cs6V3','red', 'circle')
c = Traces(Re_cs8_V3, abs_cs8V3, 'dp_cs8V3', 'green','circle')
d = Traces(Re_cs10_V3, abs_cs10V3, 'dp_cs10V3', 'blue','circle')
e = Traces(Re_cs12_V3, abs_cs12V3, 'dp_cs12V3', 'cyan','circle')


traces = [b,c,d,e]

curves(traces, 'dP x Reynolds', 'dP (Pa)')