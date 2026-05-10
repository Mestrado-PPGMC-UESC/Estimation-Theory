import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

x_dados = np.array([0,2,4,6,8])
y_dados = np.array([12.8,12.1,11.5,10.6,9.8])
confianca = np.array([5,4,3,2,1])


W = np.diag(confianca)

H = np.column_stack((
    np.ones(len(x_dados)),
    x_dados
))

x_hat = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ y_dados

tensao_estimada = x_hat[0] + x_hat[1]*x_dados

nrmse = calcular_nrmse(y_dados,tensao_estimada)

tempo_plot = np.linspace(x_dados.min(),x_dados.max(),100)
tensao_plot = x_hat[0] + x_hat[1]*tempo_plot

plt.scatter(x_dados,y_dados,label="Dados experimentais")

plt.plot(tempo_plot,tensao_plot,label=f"f(x) = {x_hat[0]} + {x_hat[1]}t  NRMSE = {nrmse:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste linear por minimos quadrados")
plt.legend()

plt.show()