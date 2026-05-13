import numpy as np


def calcular_erro_filtrado(estimativas_filtradas, distancia_real=7.8):

    posicao_final_estimativa = estimativas_filtradas[-1, 0]

    erro_estimativa = (
        distancia_real
        - posicao_final_estimativa
    )

    erro_absoluto = abs(
        erro_estimativa
    )

    erro_percentual = (
        erro_absoluto
        / distancia_real
    ) * 100

    return erro_percentual