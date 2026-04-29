import numpy as np


def gerar_estado_sem_ruido(numero_passos, estado_inicial, modelo):
    trajetoria_sem_ruido = np.zeros(numero_passos + 1)
    trajetoria_sem_ruido[0] = estado_inicial

    for passo in range(numero_passos):
        trajetoria_sem_ruido[passo + 1] = modelo.proximo_estado_sem_ruido(
            trajetoria_sem_ruido[passo]
        )

    return trajetoria_sem_ruido


def gerar_estado_com_ruido(numero_passos, estado_inicial, modelo):
    trajetoria_com_ruido = np.zeros(numero_passos + 1)
    trajetoria_com_ruido[0] = estado_inicial

    for passo in range(numero_passos):
        trajetoria_com_ruido[passo + 1] = modelo.proximo_estado(
            trajetoria_com_ruido[passo]
        )

    return trajetoria_com_ruido


def gerar_observacao(estado_real, modelo):
    numero_passos = len(estado_real)
    observacoes = np.zeros(numero_passos)

    for passo in range(numero_passos):
        observacoes[passo] = modelo.observacao(estado_real[passo])

    return observacoes


def gerar_dados(numero_passos, estado_inicial, modelo, semente_aleatoria=42):
    np.random.seed(semente_aleatoria)

    trajetoria_sem_ruido = gerar_estado_sem_ruido(
        numero_passos,
        estado_inicial,
        modelo
    )

    trajetoria_com_ruido = gerar_estado_com_ruido(
        numero_passos,
        estado_inicial,
        modelo
    )

    observacoes = gerar_observacao(
        trajetoria_com_ruido,
        modelo
    )

    return trajetoria_sem_ruido, trajetoria_com_ruido, observacoes