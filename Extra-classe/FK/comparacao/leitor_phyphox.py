import pandas as pd
import numpy as np


def carregar_dados_phyphox(caminho_arquivo):

    df = pd.read_excel(caminho_arquivo)

    tempo = df["Time (s)"].values
    aceleracao = df["Linear Acceleration y (m/s^2)"].values

    dt = np.mean(np.diff(tempo))

    velocidade_integrada = np.zeros(len(aceleracao))
    posicao_integrada = np.zeros(len(aceleracao))

    for k in range(1, len(aceleracao)):
        velocidade_integrada[k] = velocidade_integrada[k - 1] + aceleracao[k] * dt
        posicao_integrada[k] = posicao_integrada[k - 1] + velocidade_integrada[k] * dt

    medicoes = aceleracao.reshape(-1, 1)

    return tempo, medicoes, posicao_integrada, velocidade_integrada