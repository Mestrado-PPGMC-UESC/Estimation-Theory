import numpy as np


def estimar_preditor_corretor(F, H, Q, R, medicoes, numero_passos):

    x_hat = np.array([0.0, 0.0])

    P = np.array([
        [10.0, 0.0],
        [0.0, 10.0]
    ])

    estimativas_preditoras_corretoras = []

    for k in range(numero_passos):

        # Predição
        x_preditor = F @ x_hat

        P_preditor = F @ P @ F.T + Q

        # Correção
        S = H @ P_preditor @ H.T + R

        K = P_preditor @ H.T @ np.linalg.inv(S)

        inovacao = np.array([medicoes[k]]) - H @ x_preditor

        x_corrigido = x_preditor + K @ inovacao

        P_corrigido = (np.eye(2) - K @ H) @ P_preditor

        estimativas_preditoras_corretoras.append(
            x_corrigido.copy()
        )

        x_hat = x_corrigido

        P = P_corrigido

    return np.array(estimativas_preditoras_corretoras)