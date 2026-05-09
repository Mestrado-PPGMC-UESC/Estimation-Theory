import numpy as np
import matplotlib.pyplot as plt
from metrics import calcular_nrmse

# pop = x[0] = x[1]dias + x[2]dias² + x[3]dias³

pop = np.array([12,21,47,88,161,267,418])
dias = np.array([1,2,3,4,5,6,7])

H = np.column_stack((
    np.ones(len(dias)),
    dias,
    np.square(dias),
    dias**3
))

x_hat = np.linalg.inv(H.T @ H) @ H.T @ pop

pop_estimada = x_hat[0]  + x_hat[1]*dias + x_hat[2]*(dias**2) + x_hat[3]*(dias**3)

nrmse = calcular_nrmse(pop,pop_estimada)

# PLOTAGEM

x_plot = np.linspace(dias.min(),dias.max(),100)


y_plot = x_hat[0]  + x_hat[1]*x_plot + x_hat[2]*(x_plot**2) + x_hat[3]*(x_plot**3)



plt.scatter(dias,pop,label="Dados experimentais")

plt.plot(x_plot,y_plot,label=f"f(x) = {x_hat[0]:.2f} + {x_hat[1]:.2f}x  + {x_hat[2]:.2f}x² + {x_hat[3]:.2f}x³ | NRMSE = {nrmse:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste cubico por minimos quadrados")
plt.legend()

plt.show()