import matplotlib.pyplot as plt
import numpy as np

from model import ModeloPreco
from simulacao import gerar_estado_sem_ruido, gerar_estado_com_ruido
from metrics import calcular_nrmse
from config import (
    taxa_crescimento,
    volatilidade,
    passo_tempo,
    tendencia_sistematica,
    numero_passos,
    estado_inicial,
    semente_aleatoria
)


def plot_evolucao_rmse_ruido_processo(quantidade_niveis=20, delta_inicial=1e-6):

    valores_delta = np.linspace(delta_inicial, volatilidade, quantidade_niveis)
    valores_rmse = []

    for delta_atual in valores_delta:
        np.random.seed(semente_aleatoria)

        modelo = ModeloPreco(
            taxa_crescimento,
            delta_atual,
            passo_tempo,
            0.0,
            tendencia_sistematica
        )

        trajetoria_sem_ruido = gerar_estado_sem_ruido(numero_passos, estado_inicial, modelo)
        trajetoria_com_ruido = gerar_estado_com_ruido(numero_passos, estado_inicial, modelo)

        rmse_atual = calcular_nrmse(trajetoria_com_ruido, trajetoria_sem_ruido)
        valores_rmse.append(rmse_atual)

    plt.figure(figsize=(8, 5))
    plt.plot(valores_delta, valores_rmse, marker="o", linewidth=2)

    plt.title("Evolução do NRMSE com o aumento do ruído de processo")
    plt.xlabel("Ruído de processo (delta)")
    plt.ylabel("RMSE")
    plt.grid(True)

    plt.show()
