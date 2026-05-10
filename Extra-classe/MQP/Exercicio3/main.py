import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

x_dados = np.array([0,1,2,3,4,5])
y_dados = np.array([102.0,93.8,76.5,51.4,18.2,-24.6])
confianca = np.array([5,5,4,3,2,1])

W = np.diag(confianca)

H = np.column_stack((
    np.ones(len(x_dados)),
    x_dados,
    x_dados**2
))

x_hat = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ y_dados

distancia_estimada = x_hat[0] + x_hat[1]*x_dados + x_hat[2]*x_dados**2

nrmse = calcular_nrmse(y_dados,distancia_estimada)


tempo_plot = np.linspace(x_dados.min(),x_dados.max(),100)

distancia_plot = x_hat[0] + x_hat[1]*tempo_plot + x_hat[2]*tempo_plot**2

plt.scatter(x_dados,y_dados,label="Dados experimentais")

plt.plot(tempo_plot,distancia_plot,label=f"f(x) = {x_hat[0]} + {x_hat[1]}t + {x_hat[2]}t²  NRMSE = {nrmse:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste quadratico por minimos quadrados")
plt.legend()

plt.show()