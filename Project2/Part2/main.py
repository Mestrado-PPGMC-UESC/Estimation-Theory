from model import ModeloPopulacional
from analises import (
    analise_inicial_populacional,
    analise_erro_processo_populacional,
    analise_erro_observacao_populacional,
    analise_kalman_filtrado_populacional,
    analise_kalman_preditivo_populacional,
    analise_kalman_preditivo_corretivo_populacional
)

modelo = ModeloPopulacional(
    taxa_natalidade=0.03,
    taxa_mortalidade=0.02,
    ruido_processo=0.5,
    ruido_observacao=2.0
)

numero_passos = 100
estado_inicial = 100

analise_inicial_populacional(numero_passos, estado_inicial, modelo)
analise_erro_processo_populacional(numero_passos, estado_inicial, modelo)
analise_erro_observacao_populacional(numero_passos, estado_inicial, modelo)
analise_kalman_filtrado_populacional(numero_passos, estado_inicial, modelo)
analise_kalman_preditivo_populacional(numero_passos, estado_inicial, modelo)
analise_kalman_preditivo_corretivo_populacional(numero_passos, estado_inicial, modelo)