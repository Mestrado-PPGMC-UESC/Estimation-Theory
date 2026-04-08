import matplotlib.pyplot as plt
import numpy as np

def plotar_ajustes(x, y, a1, b1, a2, b2, c2,title, restrito=''):
    x_plot = np.linspace(min(x), max(x), 300)

    y_linear = a1 * x_plot + b1
    y_quadratico = a2 * x_plot**2 + b2 * x_plot + c2

    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, label='Dados')
    plt.plot(x_plot, y_linear, label=f"Ajuste Linear {restrito}")
    plt.plot(x_plot, y_quadratico, label=f"Ajuste Quadrático {restrito}")

    plt.xlabel('Renda')
    plt.ylabel('Consumo')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()