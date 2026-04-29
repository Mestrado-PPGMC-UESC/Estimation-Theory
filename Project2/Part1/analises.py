from metrics import calcular_nrmse
from plots.sistema import plot_estado,plot_medicoes,plot_erro_medicao
from plots.ruido_processo import plot_evolucao_rmse_ruido_processo
from animacoes.ruido_processo import animacao_ruido_processo
from simulacao import gerar_dados
import numpy as np
from kalman import estimativa_filtrada, estimativa_preditiva,estimativa_preditiva_corretiva
import matplotlib.pyplot as plt

def analise_inicial_ruido(trajetoria_sem_ruido, trajetoria_com_ruido):
    rmse_ruido_processo = calcular_nrmse(trajetoria_com_ruido, trajetoria_sem_ruido)

    plot_estado(trajetoria_sem_ruido, trajetoria_com_ruido)

    print(f"NRMSE entre trajetória com ruído e sem ruído: {rmse_ruido_processo:.4f}")


def analise_multiplas_sementes(numero_passos, estado_inicial, modelo,numero_iteracoes=1000):

    nrmse_medio = 0
    for semente in range(numero_iteracoes):
        trajetoria_com_ruido, trajetoria_sem_ruido, observacoes = gerar_dados(numero_passos, estado_inicial, modelo,semente)
        nrmse_medio += calcular_nrmse(trajetoria_com_ruido,trajetoria_sem_ruido)
    
    nrmse_medio = nrmse_medio/numero_iteracoes

    print(f"A media dos nrmse ao longo do tempo para {numero_iteracoes} iteracoes é {nrmse_medio}")

def analise_inicial_observacoes(trajetoria_com_ruido,observacoes):
    plot_medicoes(trajetoria_com_ruido,observacoes)
    plot_erro_medicao(trajetoria_com_ruido,observacoes)
    rmse_ruido_observação = calcular_nrmse(trajetoria_com_ruido, observacoes)
    print(f"NRMSE entre trajetória com ruído e obersação é : {rmse_ruido_observação:.4f}")

def analise_multiplas_sementes_observacao(numero_passos, estado_inicial, modelo,numero_iteracoes=1000):

    nrmse_medio = 0
    for semente in range(numero_iteracoes):
        trajetoria_com_ruido, trajetoria_sem_ruido, observacoes = gerar_dados(numero_passos, estado_inicial, modelo,semente)
        nrmse_medio += calcular_nrmse(observacoes,trajetoria_com_ruido)
    
    nrmse_medio = nrmse_medio/numero_iteracoes
    print(f"OBSERVAÇÃO: A media dos nrmse ao longo do tempo para {numero_iteracoes} iteracoes é {nrmse_medio}")


def analise_ruido_processo():

    print("\nAnálise do ruído de processo (variação de δ):\n")

    animacao_ruido_processo()
    plot_evolucao_rmse_ruido_processo()
def analise_kalman_filtrado_multiplas_sementes(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    """
    Analisa a estimativa filtrada do Filtro de Kalman.

    Calcula:
    1. NRMSE da estimativa filtrada em relação ao estado real com ruído;
    2. NRMSE da estimativa filtrada em relação à trajetória sem ruído;
    3. Média e desvio padrão desses erros para múltiplas sementes.
    """

    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_filtradas = estimativa_filtrada(
        observacoes,
        modelo,
        estado_inicial
    )

    nrmse_seed_com_ruido = calcular_nrmse(
        estimativas_filtradas,
        trajetoria_com_ruido[:len(estimativas_filtradas)]
    )

    nrmse_seed_sem_ruido = calcular_nrmse(
        estimativas_filtradas,
        trajetoria_sem_ruido[:len(estimativas_filtradas)]
    )

    print(f"NRMSE da estimativa filtrada vs com ruído para seed={seed_plot}: {nrmse_seed_com_ruido:.6f}")
    print(f"NRMSE da estimativa filtrada vs sem ruído para seed={seed_plot}: {nrmse_seed_sem_ruido:.6f}")

    k = np.arange(len(estimativas_filtradas))

    plt.figure(figsize=(12, 6))

    plt.plot(
        k,
        trajetoria_sem_ruido[:len(k)],
        label="Sem ruído (referência)",
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
        estimativas_filtradas,
        label="Estimativa filtrada (Kalman)",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Preço")
    plt.title("Filtro de Kalman - Comparação das trajetórias")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse_com_ruido = []
    lista_nrmse_sem_ruido = []

    for seed in range(numero_iteracoes):

        trajetoria_sem_ruido_i, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        estimativas_i = estimativa_filtrada(
            observacoes_i,
            modelo,
            estado_inicial
        )

        nrmse_com_ruido_i = calcular_nrmse(
            estimativas_i,
            trajetoria_com_ruido_i[:len(estimativas_i)]
        )

        nrmse_sem_ruido_i = calcular_nrmse(
            estimativas_i,
            trajetoria_sem_ruido_i[:len(estimativas_i)]
        )

        lista_nrmse_com_ruido.append(nrmse_com_ruido_i)
        lista_nrmse_sem_ruido.append(nrmse_sem_ruido_i)

    lista_nrmse_com_ruido = np.array(lista_nrmse_com_ruido)
    lista_nrmse_sem_ruido = np.array(lista_nrmse_sem_ruido)

    media_nrmse_com_ruido = np.mean(lista_nrmse_com_ruido)
    desvio_nrmse_com_ruido = np.std(lista_nrmse_com_ruido)

    media_nrmse_sem_ruido = np.mean(lista_nrmse_sem_ruido)
    desvio_nrmse_sem_ruido = np.std(lista_nrmse_sem_ruido)

    print(f"Média do NRMSE da estimativa filtrada vs com ruído para {numero_iteracoes} sementes: {media_nrmse_com_ruido:.6f}")
    print(f"Desvio padrão do NRMSE vs com ruído: {desvio_nrmse_com_ruido:.6f}")

    print(f"Média do NRMSE da estimativa filtrada vs sem ruído para {numero_iteracoes} sementes: {media_nrmse_sem_ruido:.6f}")
    print(f"Desvio padrão do NRMSE vs sem ruído: {desvio_nrmse_sem_ruido:.6f}")

    return {
        "nrmse_seed_com_ruido": nrmse_seed_com_ruido,
        "nrmse_seed_sem_ruido": nrmse_seed_sem_ruido,
        "media_nrmse_com_ruido": media_nrmse_com_ruido,
        "desvio_nrmse_com_ruido": desvio_nrmse_com_ruido,
        "media_nrmse_sem_ruido": media_nrmse_sem_ruido,
        "desvio_nrmse_sem_ruido": desvio_nrmse_sem_ruido,
        "lista_nrmse_com_ruido": lista_nrmse_com_ruido,
        "lista_nrmse_sem_ruido": lista_nrmse_sem_ruido,
        "estimativas_filtradas": estimativas_filtradas
    }


def analise_kalman_preditivo_multiplas_sementes(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_preditivas = estimativa_preditiva(
        observacoes,
        modelo,
        estado_inicial
    )

    nrmse_seed = calcular_nrmse(
        estimativas_preditivas,
        trajetoria_com_ruido[:len(estimativas_preditivas)]
    )

    print(f"NRMSE da estimativa preditiva para seed={seed_plot}: {nrmse_seed:.6f}")

    k = np.arange(len(estimativas_preditivas))

    plt.figure(figsize=(12, 6))

    plt.plot(
        k,
        trajetoria_sem_ruido[:len(k)],
        label="Sem ruído (referência)",
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
        estimativas_preditivas,
        label="Estimativa preditiva (Kalman)",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Preço")
    plt.title("Filtro de Kalman - Estimativa Preditiva")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        estimativas_i = estimativa_preditiva(
            observacoes_i,
            modelo,
            estado_inicial
        )

        nrmse_i = calcular_nrmse(
            estimativas_i,
            trajetoria_com_ruido_i[:len(estimativas_i)]
        )

        lista_nrmse.append(nrmse_i)

    lista_nrmse = np.array(lista_nrmse)

    media_nrmse = np.mean(lista_nrmse)
    desvio_nrmse = np.std(lista_nrmse)

    print(f"Média do NRMSE da estimativa preditiva para {numero_iteracoes} sementes: {media_nrmse:.6f}")
    print(f"Desvio padrão do NRMSE preditivo: {desvio_nrmse:.6f}")

    return {
        "nrmse_seed": nrmse_seed,
        "media_nrmse": media_nrmse,
        "desvio_nrmse": desvio_nrmse,
        "lista_nrmse": lista_nrmse,
        "estimativas_preditivas": estimativas_preditivas
    }


def analise_kalman_preditivo_corretivo_multiplas_sementes(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
):
    trajetoria_sem_ruido, trajetoria_com_ruido, observacoes = gerar_dados(
        numero_passos,
        estado_inicial,
        modelo,
        semente_aleatoria=seed_plot
    )

    estimativas_preditivas, estimativas_corrigidas = estimativa_preditiva_corretiva(
        observacoes,
        modelo,
        estado_inicial
    )

    nrmse_pred_seed = calcular_nrmse(
        estimativas_preditivas,
        trajetoria_com_ruido[:len(estimativas_preditivas)]
    )

    nrmse_corr_seed = calcular_nrmse(
        estimativas_corrigidas,
        trajetoria_com_ruido[:len(estimativas_corrigidas)]
    )

    print(f"NRMSE preditivo-corretivo | predição para seed={seed_plot}: {nrmse_pred_seed:.6f}")
    print(f"NRMSE preditivo-corretivo | correção para seed={seed_plot}: {nrmse_corr_seed:.6f}")

    k = np.arange(len(estimativas_corrigidas))

    plt.figure(figsize=(12, 6))


    plt.plot(
        k,
        trajetoria_sem_ruido[:len(k)],
        label="Sem ruído (referência)",
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
        label="Correção",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("Preço")
    plt.title("Filtro de Kalman - Estimativa Preditiva-Corretiva")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse_pred = []
    lista_nrmse_corr = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        pred_i, corr_i = estimativa_preditiva_corretiva(
            observacoes_i,
            modelo,
            estado_inicial
        )

        lista_nrmse_pred.append(
            calcular_nrmse(pred_i, trajetoria_com_ruido_i[:len(pred_i)])
        )

        lista_nrmse_corr.append(
            calcular_nrmse(corr_i, trajetoria_com_ruido_i[:len(corr_i)])
        )

    lista_nrmse_pred = np.array(lista_nrmse_pred)
    lista_nrmse_corr = np.array(lista_nrmse_corr)

    print(f"Média NRMSE da predição para {numero_iteracoes} sementes: {np.mean(lista_nrmse_pred):.6f}")
    print(f"Média NRMSE da correção para {numero_iteracoes} sementes: {np.mean(lista_nrmse_corr):.6f}")

    return {
        "estimativas_preditivas": estimativas_preditivas,
        "estimativas_corrigidas": estimativas_corrigidas,
        "lista_nrmse_pred": lista_nrmse_pred,
        "lista_nrmse_corr": lista_nrmse_corr,
        "media_nrmse_pred": np.mean(lista_nrmse_pred),
        "media_nrmse_corr": np.mean(lista_nrmse_corr),
    }