import numpy as np
import matplotlib.pyplot as plt


def calcular_rmse(y_real, y_estimado):

    y_real = np.asarray(y_real).flatten()
    y_estimado = np.asarray(y_estimado).flatten()

    return np.sqrt(
        np.mean(
            (y_real - y_estimado) ** 2
        )
    )


def plotar_medida_estimativa(
    tempo,
    medicoes,
    posicao_integrada,
    velocidade_integrada,
    estimativas
):

    aceleracao_medida = medicoes.flatten()

    posicao_kalman = estimativas[:, 0]
    velocidade_kalman = estimativas[:, 1]
    aceleracao_kalman = estimativas[:, 2]

    rmse_posicao = calcular_rmse(
        posicao_integrada,
        posicao_kalman
    )

    rmse_velocidade = calcular_rmse(
        velocidade_integrada,
        velocidade_kalman
    )

    rmse_aceleracao = calcular_rmse(
        aceleracao_medida,
        aceleracao_kalman
    )

    # -----------------------------
    # Posição
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        tempo,
        posicao_integrada,
        label="Posição integrada"
    )

    plt.plot(
        tempo,
        posicao_kalman,
        label="Posição estimada - Kalman"
    )

    plt.title(
        f"Posição | RMSE = {rmse_posicao:.6f}"
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição (m)")

    plt.grid()
    plt.legend()
    plt.show()

    # -----------------------------
    # Velocidade
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        tempo,
        velocidade_integrada,
        label="Velocidade integrada"
    )

    plt.plot(
        tempo,
        velocidade_kalman,
        label="Velocidade estimada - Kalman"
    )

    plt.title(
        f"Velocidade | RMSE = {rmse_velocidade:.6f}"
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade (m/s)")

    plt.grid()
    plt.legend()
    plt.show()

    # -----------------------------
    # Aceleração
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        tempo,
        aceleracao_medida,
        label="Aceleração medida"
    )

    plt.plot(
        tempo,
        aceleracao_kalman,
        label="Aceleração estimada - Kalman"
    )

    plt.title(
        f"Aceleração | RMSE = {rmse_aceleracao:.6f}"
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel("Aceleração (m/s²)")

    plt.grid()
    plt.legend()
    plt.show()

    print(f"RMSE posição: {rmse_posicao:.6f}")
    print(f"RMSE velocidade: {rmse_velocidade:.6f}")
    print(f"RMSE aceleração: {rmse_aceleracao:.6f}")