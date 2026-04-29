import matplotlib.pyplot as plt
import numpy as np
from metrics import calcular_rmse

def plot_estado(trajetoria_sem_ruido, trajetoria_com_ruido):

    k = np.arange(len(trajetoria_sem_ruido))
    erro_entre_trajetorias = trajetoria_com_ruido - trajetoria_sem_ruido

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k, trajetoria_sem_ruido, label="Sem ruído", linewidth=2)
    ax1.plot(k, trajetoria_com_ruido, label="Com ruído", alpha=0.7)

    ax1.set_title("Evolução do preço")
    ax1.set_xlabel("Passo de tempo (k)")
    ax1.set_ylabel("Preço")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(k, erro_entre_trajetorias, label="Erro", linewidth=2)
    ax2.axhline(0, linestyle="--", linewidth=1)

    ax2.set_title("Erro")
    ax2.set_xlabel("Passo de tempo (k)")
    ax2.set_ylabel("Erro")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_observacao(p_real, y):
    """
    Plota o estado real e a observação.

    Parâmetros:
        p_real: estado com ruído
        y: observações
    """

    k = np.arange(len(y))

    plt.figure(figsize=(10, 5))
    plt.plot(k, p_real, label="Estado real", linewidth=2)
    plt.scatter(k, y, label="Observação", s=10, alpha=0.7)

    plt.title("Equação de Observação")
    plt.xlabel("k")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()

    plt.show()


def plot_completo(p_real, y, estimado=None):
    """
    Plota estado, observação e estimativa (opcional).
    """

    k = np.arange(len(y))

    plt.figure(figsize=(10, 5))
    plt.plot(k, p_real, label="Estado real", linewidth=2)
    plt.scatter(k, y, label="Observação", s=10, alpha=0.6)

    if estimado is not None:
        plt.plot(k, estimado, label="Estimativa Kalman", linestyle="--")

    plt.title("Comparação geral")
    plt.xlabel("k")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()

    plt.show()


def plot_filtrado(p_sem_ruido, p_com_ruido, x_filt):
    """
    Plota a trajetória sem ruído, a trajetória com ruído
    e a estimativa filtrada.

    Parâmetros:
        p_sem_ruido: trajetória determinística
        p_com_ruido: trajetória com ruído de processo
        x_filt: estimativa filtrada pelo filtro de Kalman
    """

    k = np.arange(len(p_sem_ruido))

    plt.figure(figsize=(10, 5))
    plt.plot(k, p_sem_ruido, label="Sem ruído", linewidth=2)
    plt.plot(k, p_com_ruido, label="Com ruído", linewidth=2, alpha=0.8)
    plt.plot(k, x_filt, label="Estimativa filtrada", linestyle="--", linewidth=2)

    plt.title("Comparação entre estado sem ruído, com ruído e estimativa filtrada")
    plt.xlabel("k")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid()

    plt.show()