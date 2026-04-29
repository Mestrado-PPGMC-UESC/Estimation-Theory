from fusion import agregar_observacoes_choquet
from simulacao import gerar_dados
from kalman import estimativa_preditiva_corretiva
from metrics import calcular_nrmse


from model import ModeloPopulacional
from analises import (
    analise_inicial_populacional,
    analise_erro_processo_populacional,
    analise_erro_observacao_populacional,
    analise_kalman_preditivo_corretivo_populacional,
    analise_choquet_kalman_populacional
)


modelo = ModeloPopulacional(
    taxa_natalidade=0.03,
    taxa_mortalidade=0.02,
    ruido_processo=0.5,
    ruidos_observacao=[1.0, 1.5, 8.0]
)

modelo_choquet = ModeloPopulacional(
    taxa_natalidade=0.03,
    taxa_mortalidade=0.02,
    ruido_processo=0.5,
    ruido_observacao=1.0
)

numero_passos = 100
estado_inicial = 100

medida_mu = {
    (0,): 0.75,   # sensor 1 bom
    (1,): 0.70,   # sensor 2 bom
    (2,): 0.15,   # sensor 3 ruim

    (0, 1): 0.95,
    (0, 2): 0.80,
    (1, 2): 0.75,

    (0, 1, 2): 1.00
}

analise_inicial_populacional(numero_passos, estado_inicial, modelo)

analise_erro_processo_populacional(
    numero_passos,
    estado_inicial,
    modelo
)

analise_erro_observacao_populacional(
    numero_passos,
    estado_inicial,
    modelo
)

print("\n===== Kalman com múltiplos sensores =====")
analise_kalman_preditivo_corretivo_populacional(
    numero_passos,
    estado_inicial,
    modelo
)

print("\n===== Choquet + Kalman =====")
analise_choquet_kalman_populacional(
    numero_passos,
    estado_inicial,
    modelo,
    modelo_choquet,
    medida_mu
)