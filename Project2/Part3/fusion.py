import numpy as np


def choquet_3_sensores(valores, medida_mu):
    """
    Calcula a integral de Choquet para 3 sensores.

    valores: [z1, z2, z3]
    medida_mu: dicionário com a medida fuzzy dos subconjuntos.
    """

    valores = np.array(valores).flatten()

    indices_ordenados = np.argsort(valores)
    valores_ordenados = valores[indices_ordenados]

    resultado = 0.0
    valor_anterior = 0.0

    for i in range(len(valores)):
        conjunto = tuple(sorted(indices_ordenados[i:]))

        resultado += (
            valores_ordenados[i] - valor_anterior
        ) * medida_mu[conjunto]

        valor_anterior = valores_ordenados[i]

    return resultado


def agregar_observacoes_choquet(observacoes, medida_mu):
    """
    Aplica Choquet em todas as medições.

    Entrada:
        observacoes: formato (N, 3, 1)

    Saída:
        observacoes_agregadas: formato (N,)
    """

    numero_passos = len(observacoes)
    observacoes_agregadas = np.zeros(numero_passos)

    for k in range(numero_passos):
        sensores_k = observacoes[k].flatten()

        observacoes_agregadas[k] = choquet_3_sensores(
            sensores_k,
            medida_mu
        )

    return observacoes_agregadas