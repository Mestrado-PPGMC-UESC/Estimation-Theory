import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

h_dados = np.array([192,180,150,115,72])
t_dados = np.array([1,2,3,4,5])

# S = So + vot + gt²/2

H = np.column_stack((
    np.ones(len(t_dados)),
    t_dados,
    np.square(t_dados)
))

x_hat = np.linalg.inv(H.T @ H) @ H.T @ h_dados

So = x_hat[0]
vo = x_hat[1]
g = -2*x_hat[2]

print(f"So = {So:.4f}")
print(f"vo = {vo:.4f}")
print(f"g = {g:.4f}")

h_estimado = So + vo*t_dados - g*np.square(t_dados)/2

nrmse= calcular_nrmse(h_dados,h_estimado)

#PLOTAGEM

x_plot = np.linspace(t_dados.min(),t_dados.max(),100)

y_plot = So + vo*x_plot - g*np.square(x_plot)/2

plt.scatter(t_dados,h_dados,label="Dados de observação")

plt.plot(x_plot,y_plot,label=f"f(x) = {So:2f} + {vo:2f}t  {g:2f}t²/2 | NRMSE = {nrmse:.2f}")
plt.xlabel("Tempo")
plt.ylabel("Altura")
plt.title("Ajuste Quadratico por minimos quadraticos")
plt.legend()
plt.show()
