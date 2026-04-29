import numpy as np
import matplotlib.pyplot as plt

from simulacao import gerar_dados
from metrics import calcular_nrmse
from kalman import estimativa_preditiva_corretiva
from fusion import agregar_observacoes_choquet

def obter_observacao_media(observacoes):
    """
    Retorna uma série 1D para plotar/medir observações.

    - 1 sensor: retorna observacoes normal.
    - multissensor: retorna a média dos sensores em cada instante.
    """

    observacoes = np.array(observacoes)

    if observacoes.ndim == 3:
        return observacoes[:, :, 0].mean(axis=1)

    return observacoes


def obter_sensor(observacoes, indice_sensor):
    """
    Retorna a série temporal de um sensor específico.
    """

    observacoes = np.array(observacoes)

    if observacoes.ndim == 3:
        return observacoes[:, indice_sensor, 0]

    return observacoes


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
    plt.plot(k, trajetoria_com_ruido, label="Com ruído (estado real)", alpha=0.8)

    if np.array(observacoes).ndim == 3:
        quantidade_sensores = observacoes.shape[1]

        for i in range(quantidade_sensores):
            plt.plot(
                k,
                obter_sensor(observacoes, i),
                label=f"Sensor {i + 1}",
                alpha=0.4
            )
    else:
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

    print(f"NRMSE processo para seed={seed_plot}: {nrmse_processo:.6f}")

    if np.array(observacoes).ndim == 3:
        quantidade_sensores = observacoes.shape[1]

        for i in range(quantidade_sensores):
            nrmse_sensor = calcular_nrmse(
                obter_sensor(observacoes, i),
                trajetoria_com_ruido
            )

            print(
                f"NRMSE observação sensor {i + 1} para seed={seed_plot}: "
                f"{nrmse_sensor:.6f}"
            )

        observacao_media = obter_observacao_media(observacoes)

        nrmse_media = calcular_nrmse(
            observacao_media,
            trajetoria_com_ruido
        )

        print(
            f"NRMSE observação média dos sensores para seed={seed_plot}: "
            f"{nrmse_media:.6f}"
        )

    else:
        nrmse_observacao = calcular_nrmse(
            observacoes,
            trajetoria_com_ruido
        )

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
    listas_nrmse_sensores = None
    lista_nrmse_media = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido, observacoes = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        observacoes = np.array(observacoes)

        if observacoes.ndim == 3:
            quantidade_sensores = observacoes.shape[1]

            if listas_nrmse_sensores is None:
                listas_nrmse_sensores = [[] for _ in range(quantidade_sensores)]

            for i in range(quantidade_sensores):
                nrmse_sensor = calcular_nrmse(
                    obter_sensor(observacoes, i),
                    trajetoria_com_ruido
                )

                listas_nrmse_sensores[i].append(nrmse_sensor)

            observacao_media = obter_observacao_media(observacoes)

            lista_nrmse_media.append(
                calcular_nrmse(
                    observacao_media,
                    trajetoria_com_ruido
                )
            )

        else:
            if listas_nrmse_sensores is None:
                listas_nrmse_sensores = [[]]

            nrmse = calcular_nrmse(
                observacoes,
                trajetoria_com_ruido
            )

            listas_nrmse_sensores[0].append(nrmse)

    resultados = {}

    for i, lista in enumerate(listas_nrmse_sensores):
        lista = np.array(lista)

        media = np.mean(lista)
        desvio = np.std(lista)

        print(
            f"Média NRMSE do ruído de observação sensor {i + 1} "
            f"para {numero_iteracoes} sementes: {media:.6f}"
        )
        print(
            f"Desvio padrão NRMSE do ruído de observação sensor {i + 1}: "
            f"{desvio:.6f}"
        )

        resultados[f"media_nrmse_sensor_{i + 1}"] = media
        resultados[f"desvio_nrmse_sensor_{i + 1}"] = desvio
        resultados[f"lista_nrmse_sensor_{i + 1}"] = lista

    if len(lista_nrmse_media) > 0:
        lista_nrmse_media = np.array(lista_nrmse_media)

        media = np.mean(lista_nrmse_media)
        desvio = np.std(lista_nrmse_media)

        print(
            f"Média NRMSE da média dos sensores para {numero_iteracoes} sementes: "
            f"{media:.6f}"
        )
        print(
            f"Desvio padrão NRMSE da média dos sensores: {desvio:.6f}"
        )

        resultados["media_nrmse_media_sensores"] = media
        resultados["desvio_nrmse_media_sensores"] = desvio
        resultados["lista_nrmse_media_sensores"] = lista_nrmse_media

    return resultados


def analise_kalman_preditivo_corretivo_populacional(
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

    if np.array(observacoes).ndim == 3:
        quantidade_sensores = observacoes.shape[1]

        for i in range(quantidade_sensores):
            plt.plot(
                k,
                obter_sensor(observacoes, i)[:len(k)],
                label=f"Sensor {i + 1}",
                alpha=0.25
            )

    else:
        plt.plot(
            k,
            observacoes[:len(k)],
            label="Observações",
            alpha=0.4
        )

    plt.plot(
        k,
        estimativas_corrigidas,
        label="Estimativa Kalman",
        linewidth=2
    )

    plt.xlabel("Passo de tempo (k)")
    plt.ylabel("População")
    plt.title("Filtro de Kalman - Fusão de Medições")
    plt.legend()
    plt.grid(True)
    plt.show()

    lista_nrmse_kalman = []

    for seed in range(numero_iteracoes):

        _, trajetoria_com_ruido_i, observacoes_i = gerar_dados(
            numero_passos,
            estado_inicial,
            modelo,
            semente_aleatoria=seed
        )

        _, estimativas_corrigidas_i = estimativa_preditiva_corretiva(
            observacoes_i,
            modelo,
            estado_inicial
        )

        lista_nrmse_kalman.append(
            calcular_nrmse(
                estimativas_corrigidas_i,
                trajetoria_com_ruido_i[:len(estimativas_corrigidas_i)]
            )
        )

    lista_nrmse_kalman = np.array(lista_nrmse_kalman)

    media_kalman = np.mean(lista_nrmse_kalman)

    print(
        f"Média NRMSE da fusão por Kalman para {numero_iteracoes} sementes: "
        f"{media_kalman:.6f}"
    )


    return {
        "media_nrmse_kalman": media_kalman,
        "lista_nrmse_kalman": lista_nrmse_kalman,
        "estimativas_preditivas": estimativas_preditivas,
        "estimativas_corrigidas": estimativas_corrigidas
    }
def analise_choquet_kalman_populacional(
    numero_passos,
    estado_inicial,
    modelo_multissensor,
    modelo_sensor_unico,
    medida_mu,
    seed_plot=42
):
    import matplotlib.pyplot as plt
    import numpy as np
    from simulacao import gerar_dados
    from kalman import estimativa_preditiva_corretiva
    from metrics import calcular_nrmse
    from fusion import agregar_observacoes_choquet

    # ============================================================
    # Gerar dados
    # ============================================================
    traj_sem, traj_real, observacoes = gerar_dados(
        numero_passos,
        estado_inicial,
        modelo_multissensor,
        semente_aleatoria=seed_plot
    )

    # ============================================================
    # Aplicar Choquet
    # ============================================================
    observacoes_choquet = agregar_observacoes_choquet(
        observacoes,
        medida_mu
    )

    # ============================================================
    # Rodar Kalman
    # ============================================================
    _, estimativas = estimativa_preditiva_corretiva(
        observacoes_choquet,
        modelo_sensor_unico,
        estado_inicial
    )

    # ============================================================
    # GRÁFICO
    # ============================================================
    k = np.arange(len(estimativas))

    plt.figure(figsize=(12, 6))

    plt.plot(k, traj_sem[:len(k)], label="Sem ruído", linewidth=2)
    plt.plot(k, traj_real[:len(k)], label="Estado real", alpha=0.8)

    # sensores
    for i in range(observacoes.shape[1]):
        plt.plot(
            k,
            observacoes[:, i, 0][:len(k)],
            label=f"Sensor {i+1}",
            alpha=0.25
        )

    plt.plot(
        k,
        observacoes_choquet[:len(k)],
        label="Choquet",
        linewidth=2,
        linestyle="--"
    )

    plt.plot(
        k,
        estimativas,
        label="Kalman (Choquet)",
        linewidth=2
    )

    plt.title("Choquet + Filtro de Kalman")
    plt.xlabel("Tempo")
    plt.ylabel("População")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================================================
    # ERRO
    # ============================================================
    nrmse = calcular_nrmse(
        estimativas,
        traj_real[:len(estimativas)]
    )

    print(f"NRMSE Choquet + Kalman: {nrmse:.6f}")

    return {
        "nrmse": nrmse,
        "estimativas": estimativas,
        "observacoes_choquet": observacoes_choquet
    }