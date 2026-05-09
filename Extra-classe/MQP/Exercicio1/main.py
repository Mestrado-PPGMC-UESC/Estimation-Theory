import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

temperatura_dados = np.array([20.5,24.8,31.2,35.7,42.6])
tempo_dados = np.array([0,1,2,3,4])
confianca = np.array([5,4,2,1,1])

H = np.column_stack((
    np.ones(len(tempo_dados)),
    tempo_dados
))

W = np.diag(confianca)

x_hat = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ temperatura_dados

temperatura_estimada = x_hat[0] + x_hat[1]*tempo_dados

nrmse = calcular_nrmse(temperatura_dados,temperatura_estimada)

tempo_plot = np.linspace(tempo_dados.min(),tempo_dados.max(),100)

temperatura_plot = x_hat[0] + x_hat[1]*tempo_plot

plt.scatter(tempo_dados,temperatura_dados,label="Dados experimentais")

plt.plot(tempo_plot,temperatura_plot,label=f"f(x) = {x_hat[0]} + {x_hat[1]}t")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste linear por minimos quadrados")
plt.legend()

plt.show()