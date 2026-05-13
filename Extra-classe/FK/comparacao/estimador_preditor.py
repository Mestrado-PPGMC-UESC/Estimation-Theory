import numpy as np


def estimar_preditor(F, H, Q, R, medicoes, numero_passos):

    x_hat = np.array([0.0, 0.0])

    P = np.array([
        [10.0, 0.0],
        [0.0, 10.0]
    ])

    estimativas_preditoras = [x_hat.copy()]

    for k in range(numero_passos - 1):

        S = R + H @ P @ H.T

        K = P @ H.T @ np.linalg.inv(S)

        inovacao = np.array([medicoes[k]]) - H @ x_hat

        x_filtrado = x_hat + K @ inovacao

        x_preditor = F @ x_filtrado

        P_filtrado = P - K @ H @ P

        P_preditor = Q + F @ P_filtrado @ F.T

        estimativas_preditoras.append(x_preditor.copy())

        x_hat = x_preditor

        P = P_preditor

    return np.array(estimativas_preditoras)