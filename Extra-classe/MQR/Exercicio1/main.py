import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

x_dados = np.array([0,1,2,3,4,5,6])

y_dados = np.array([
    5.1,
    10.8,
    13.2,
    39.7,
    48.4,
    112.6,
    141.3
])

H = np.column_stack((
    np.ones(len(x_dados)),
    x_dados,
    x_dados**2,
    x_dados**3
))

lamb = 0.00001
Q = lamb * np.eye(H.shape[1])

x_hat = np.linalg.inv(Q + H.T @ H) @ H.T @ y_dados

populacao_estimada = x_hat[0] + x_hat[1]*x_dados + x_hat[2]*x_dados**2 + x_hat[3]*x_dados**3


nrmse = calcular_nrmse(y_dados,populacao_estimada)


tempo_plot = np.linspace(x_dados.min(),x_dados.max(),100)

populacao_plot = x_hat[0] + x_hat[1]*tempo_plot + x_hat[2]*tempo_plot**2 + x_hat[3]*tempo_plot**3

plt.scatter(x_dados,y_dados,label="Dados experimentais")

plt.plot(tempo_plot,populacao_plot,label=f"f(x) = {x_hat[0]} + {x_hat[1]}t + {x_hat[2]}t²  NRMSE = {nrmse:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste cubico por minimos quadrados")
plt.legend()

plt.show()
