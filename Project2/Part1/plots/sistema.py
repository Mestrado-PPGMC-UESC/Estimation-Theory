import matplotlib.pyplot as plt
import numpy as np

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
    ax2.set_ylabel("Erro ")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_ruido_processo(trajetoria_sem_ruido, trajetoria_com_ruido):

    eixo_tempo = np.arange(len(trajetoria_sem_ruido))
    erro_absoluto = np.abs(trajetoria_com_ruido - trajetoria_sem_ruido)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(eixo_tempo, trajetoria_sem_ruido, label="Sem ruído", linewidth=2)
    ax1.plot(eixo_tempo, trajetoria_com_ruido, label="Com ruído", alpha=0.7)

    ax1.set_title("Evolução do preço")
    ax1.set_xlabel("Passo de tempo (k)")
    ax1.set_ylabel("Preço")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(eixo_tempo, erro_absoluto, label="Erro absoluto", linewidth=2)

    ax2.set_title("Erro absoluto entre as trajetórias")
    ax2.set_xlabel("Passo de tempo (k)")
    ax2.set_ylabel("Erro")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_medicoes(trajetoria_com_ruido, observacoes):

    k = np.arange(len(trajetoria_com_ruido))

    plt.figure(figsize=(10, 5))

    plt.plot(k, trajetoria_com_ruido, label="Estado real (com ruído)", linewidth=2)
    plt.plot(k, observacoes, label="Medições", alpha=0.7)

    plt.title("Estado real vs medições")
    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_erro_medicao(trajetoria_com_ruido, observacoes):

    k = np.arange(len(trajetoria_com_ruido))
    erro_medicao = observacoes - trajetoria_com_ruido

    plt.figure(figsize=(10, 5))

    plt.plot(k, erro_medicao, label="Erro de medição", linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.title("Erro das medições")
    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Erro")
    plt.legend()
    plt.grid(True)

    plt.show()