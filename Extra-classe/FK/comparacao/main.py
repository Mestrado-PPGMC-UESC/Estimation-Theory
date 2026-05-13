import numpy as np
import matplotlib.pyplot as plt

from leitor_phyphox import carregar_dados_phyphox
from parametros import obter_parametros
from estimador_filtrado import estimar_filtrado
from plotter import plotar_medida_estimativa
from plano_q_r import montar_plano_q_r


if __name__ == "__main__":

    tempo, medicoes, posicao_integrada, velocidade_integrada = carregar_dados_phyphox("dados_phyphox.xls")

    numero_passos = len(medicoes)
    distancia_real = 7.8

    q_valor = 0.01
    r_valor = 0.5

    dt, F, H, Q, R = obter_parametros(
        tempo=tempo,
        q_valor=q_valor,
        r_valor=r_valor
    )

    estimativas_filtradas, covariancias_filtradas = estimar_filtrado(
        F=F,
        H=H,
        Q=Q,
        R=R,
        medicoes=medicoes,
        numero_passos=numero_passos
    )

    posicao_estimativa = estimativas_filtradas[:, 0]

    distancia_estimada = posicao_estimativa[-1].item()

    erro_percentual = (
        abs(distancia_estimada - distancia_real)
        / distancia_real
    ) * 100

    print(f"Distância real: {distancia_real:.4f} m")
    print(f"Distância estimada Kalman: {distancia_estimada:.4f} m")
    print(f"Erro percentual inicial: {erro_percentual:.2f} %")

    valores_q, valores_r, matriz_erro, melhor_q, melhor_r, menor_erro = montar_plano_q_r(
        F=F,
        H=H,
        medicoes=medicoes
    )

    print("\nMelhores parâmetros encontrados")
    print("--------------------------------")
    print(f"Q ótimo: {melhor_q:.4f}")
    print(f"R ótimo: {melhor_r:.4f}")
    print(f"Menor erro encontrado: {menor_erro:.6f} %")

    # Reexecuta com Q e R ótimos
    _, _, Q_otimo, R_otimo = obter_parametros(
        tempo=tempo,
        q_valor=melhor_q,
        r_valor=melhor_r
    )[1:]

    estimativas_filtradas_otimas, _ = estimar_filtrado(
        F=F,
        H=H,
        Q=Q_otimo,
        R=R_otimo,
        medicoes=medicoes,
        numero_passos=numero_passos
    )

    plotar_medida_estimativa(
        tempo=tempo,
        medicoes=medicoes,
        posicao_integrada=posicao_integrada,
        velocidade_integrada=velocidade_integrada,
        estimativas=estimativas_filtradas_otimas
    )