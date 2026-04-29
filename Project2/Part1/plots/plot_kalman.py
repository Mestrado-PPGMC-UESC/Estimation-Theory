import matplotlib.pyplot as plt
import numpy as np


def plot_kalman_comparacao(
    trajetoria_sem_ruido,
    trajetoria_com_ruido,
    estimativa_filtrada
):
    k = np.arange(len(estimativa_filtrada))

    plt.figure(figsize=(12, 6))

    plt.plot(
        k,
        trajetoria_sem_ruido[:len(k)],
        label="Sem ruído (referência)",
        linewidth=2
    )

    plt.plot(
        k,
        trajetoria_com_ruido[:len(k)],
        label="Com ruído (estado real)",
        alpha=0.7
    )

    plt.plot(
        k,
        estimativa_filtrada,
        label="Estimativa filtrada (Kalman)",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Preço")
    plt.title("Filtro de Kalman - Comparação das trajetórias")
    plt.legend()
    plt.grid(True)

    plt.show()