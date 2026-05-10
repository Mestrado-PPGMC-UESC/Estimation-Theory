import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

x_dados = np.array([0,1,2,3])
y_dados = np.array([1,2,4,8])

H = np.column_stack((
    np.ones(len(x_dados)),
    x_dados
))

x_hat = np.linalg.inv(H.T @ H) @ H.T @ y_dados;

a = x_hat[0]
b = x_hat[1]

print(f"a = {a:.4f}")
print(f"b = {b:.4f}")

y_estimado = a + b*x_dados

nrmse= calcular_nrmse(y_dados,y_estimado)

# PLOTAGEM

x_plot = np.linspace(x_dados.min(),x_dados.max(),100)


y_plot = a + b*x_plot



plt.scatter(x_dados,y_dados,label="Dados experimentais")

plt.plot(x_plot,y_plot,label=f"f(x) = {a:.2f} + {b:.2f}x | NRMSE = {nrmse:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste linear por minimos quadrados")
plt.legend()

plt.show()