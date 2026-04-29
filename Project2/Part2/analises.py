import numpy as np
import matplotlib.pyplot as plt

from simulacao import gerar_dados
from metrics import calcular_nrmse
from kalman import (
    estimativa_filtrada,
    estimativa_preditiva,
    estimativa_preditiva_corretiva
)


def analise_inicial_populacional(
    numero_passos,
    estado_inicial,
    modelo,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )



    k = np.arange(len(trajetoria_sem_ruido))

    plt.figure(figsize=(12, 6))
    plt.plot(k, trajetoria_sem_ruido, label="Sem ruído", linewidth=2)
    plt.plot(k, trajetoria_com_ruido, label="Com ruído", alpha=0.8)
    plt.plot(k, observacoes, label="Observações", alpha=0.5)

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("População")
    plt.title("Modelo Populacional - Simulação")
    plt.legend()
    plt.grid(True)
    plt.show()

    nrmse_processo = calcular_nrmse(
        trajetoria_com_ruido,
        trajetoria_sem_ruido
    )

    nrmse_observacao = calcular_nrmse(
        observacoes,
        trajetoria_com_ruido
    )

    print(f"NRMSE processo para seed={seed_plot}: {nrmse_processo:.6f}")
    print(f"NRMSE observação para seed={seed_plot}: {nrmse_observacao:.12f}")

    return trajetoria_sem_ruido, trajetoria_com_ruido, observacoes


def analise_erro_processo_populacional(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000
):
    lista_nrmse = []

    for seed in range(numero_iteracoes):

        trajetoria_sem_ruido, trajetoria_com_ruido, _ = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        nrmse = calcular_nrmse(
            trajetoria_com_ruido,
            trajetoria_sem_ruido
        )

        lista_nrmse.append(nrmse)

    lista_nrmse = np.array(lista_nrmse)

    media = np.mean(lista_nrmse)
    desvio = np.std(lista_nrmse)

    print(f"Média NRMSE do ruído de processo para {numero_iteracoes} sementes: {media:.6f}")
    print(f"Desvio padrão NRMSE do ruído de processo: {desvio:.6f}")

    return {
        "media_nrmse_processo": media,
        "desvio_nrmse_processo": desvio,
        "lista_nrmse_processo": lista_nrmse
    }


def analise_erro_observacao_populacional(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000
):
    lista_nrmse = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido, observacoes = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        nrmse = calcular_nrmse(
            observacoes,
            trajetoria_com_ruido
        )

        lista_nrmse.append(nrmse)

    lista_nrmse = np.array(lista_nrmse)

    media = np.mean(lista_nrmse)
    desvio = np.std(lista_nrmse)

    print(f"Média NRMSE do ruído de observação para {numero_iteracoes} sementes: {media:.6f}")
    print(f"Desvio padrão NRMSE do ruído de observação: {desvio:.6f}")

    return {
        "media_nrmse_observacao": media,
        "desvio_nrmse_observacao": desvio,
        "lista_nrmse_observacao": lista_nrmse
    }

def analise_kalman_filtrado_populacional(
    numero_passos,
    estado_real_inicial,
    estado_estimado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_real_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_filtradas = estimativa_filtrada(
        observacoes,
        modelo,
        estado_estimado_inicial
    )

    k = np.arange(len(estimativas_filtradas))

    plt.figure(figsize=(12, 6))
    plt.plot(k, trajetoria_sem_ruido[:len(k)], label="Sem ruído", linewidth=2)
    plt.plot(k, trajetoria_com_ruido[:len(k)], label="Com ruído", alpha=0.8)
    plt.plot(k, estimativas_filtradas, label="Estimativa filtrada", linewidth=2)

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("População")
    plt.title("Filtro de Kalman - Estimativa Filtrada")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_real_inicial,
            modelo,
            semente_aleatoria=seed
        )

        estimativas_i = estimativa_filtrada(
            observacoes_i,
            modelo,
            estado_estimado_inicial
        )

        nrmse = calcular_nrmse(
            estimativas_i,
            trajetoria_com_ruido_i[:len(estimativas_i)]
        )

        lista_nrmse.append(nrmse)

    lista_nrmse = np.array(lista_nrmse)

    media = np.mean(lista_nrmse)
    desvio = np.std(lista_nrmse)

    print(f"Média NRMSE da estimativa filtrada para {numero_iteracoes} sementes: {media:.6f}")
    print(f"Desvio padrão NRMSE da estimativa filtrada: {desvio:.6f}")

    return {
        "media_nrmse_filtrada": media,
        "desvio_nrmse_filtrada": desvio,
        "lista_nrmse_filtrada": lista_nrmse,
        "estimativas_filtradas": estimativas_filtradas
    }

def analise_kalman_preditivo_populacional(
    numero_passos,
    estado_real_inicial,
    estado_estimado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_real_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_preditivas = estimativa_preditiva(
        observacoes,
        modelo,
        estado_estimado_inicial
    )

    k = np.arange(len(estimativas_preditivas))

    plt.figure(figsize=(12, 6))
    plt.plot(k, trajetoria_sem_ruido[:len(k)], label="Sem ruído", linewidth=2)
    plt.plot(k, trajetoria_com_ruido[:len(k)], label="Com ruído", alpha=0.8)
    plt.plot(k, estimativas_preditivas, label="Estimativa preditiva", linewidth=2)

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("População")
    plt.title("Filtro de Kalman - Estimativa Preditiva")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_real_inicial,
            modelo,
            semente_aleatoria=seed
        )

        estimativas_i = estimativa_preditiva(
            observacoes_i,
            modelo,
            estado_estimado_inicial
        )

        nrmse = calcular_nrmse(
            estimativas_i,
            trajetoria_com_ruido_i[:len(estimativas_i)]
        )

        lista_nrmse.append(nrmse)

    lista_nrmse = np.array(lista_nrmse)

    media = np.mean(lista_nrmse)
    desvio = np.std(lista_nrmse)

    print(f"Média NRMSE da estimativa preditiva para {numero_iteracoes} sementes: {media:.6f}")
    print(f"Desvio padrão NRMSE da estimativa preditiva: {desvio:.6f}")

    return {
        "media_nrmse_preditiva": media,
        "desvio_nrmse_preditiva": desvio,
        "lista_nrmse_preditiva": lista_nrmse,
        "estimativas_preditivas": estimativas_preditivas
    }

def analise_kalman_preditivo_corretivo_populacional(
    numero_passos,
    estado_real_inicial,
    estado_estimado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_real_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_preditivas, estimativas_corrigidas = estimativa_preditiva_corretiva(
        observacoes,
        modelo,
        estado_estimado_inicial
    )

    k = np.arange(len(estimativas_corrigidas))

    plt.figure(figsize=(12, 6))

    plt.plot(
        k,
        trajetoria_sem_ruido[:len(k)],
        label="Sem ruído",
        linewidth=2
    )

    plt.plot(
        k,
        trajetoria_com_ruido[:len(k)],
        label="Com ruído (estado real)",
        alpha=0.8
    )

    plt.plot(
        k,
        estimativas_corrigidas,
        label="Estimativa corrigida",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("População")
    plt.title("Filtro de Kalman - Estimativa Preditiva-Corretiva")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse_corr = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_real_inicial,
            modelo,
            semente_aleatoria=seed
        )

        _, corr_i = estimativa_preditiva_corretiva(
            observacoes_i,
            modelo,
            estado_estimado_inicial
        )

        lista_nrmse_corr.append(
            calcular_nrmse(corr_i, trajetoria_com_ruido_i[:len(corr_i)])
        )

    lista_nrmse_corr = np.array(lista_nrmse_corr)

    media_corr = np.mean(lista_nrmse_corr)
    desvio_corr = np.std(lista_nrmse_corr)

    print(f"Média NRMSE da preditiva-corretiva para {numero_iteracoes} sementes: {media_corr:.6f}")
    print(f"Desvio padrão NRMSE da preditiva-corretiva: {desvio_corr:.6f}")

    return {
        "media_nrmse_corr": media_corr,
        "desvio_nrmse_corr": desvio_corr,
        "lista_nrmse_corr": lista_nrmse_corr,
        "estimativas_preditivas": estimativas_preditivas,
        "estimativas_corrigidas": estimativas_corrigidas
    }