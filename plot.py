import numpy as np
import matplotlib.pyplot as plt

# Gerando os dados
x = np.linspace(0, 6.28, 100)

cos = np.cos(x)
sen = np.sin(x)
tg = np.tan(x)

# Criando o gráfico
plt.plot(x, cos, label='cos(x)')
plt.plot(x, sen, label='sin(x)')
plt.plot(x, tg, label='tan(x)')

# Adicionando título e rótulos aos eixos
plt.title('Trigonometric Functions')
plt.xlabel('X (rad)')
plt.ylabel('Amplitude')

# Adicionando legenda
plt.legend()

# Definindo os limites dos eixos para evitar infinidades na tangente
plt.ylim(-2, 2)

# Mostrando o gráfico
plt.show()

