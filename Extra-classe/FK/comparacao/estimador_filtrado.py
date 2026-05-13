import numpy as np


def estimar_filtrado(F, H, Q, R, medicoes, numero_passos):

    P0 = 10 * np.eye(3)

    P = np.linalg.inv(
        np.linalg.inv(P0) + H.T @ np.linalg.inv(R) @ H
    )

    x_hat = P @ H.T @ np.linalg.inv(R) @ np.array([
        [medicoes[0].item()]
    ])

    estimativas_filtradas = [x_hat.flatten()]
    covariancias_filtradas = [P.copy()]

    for k in range(1, numero_passos):

        P_pred = Q + F @ P @ F.T

        P_filt = np.linalg.inv(
            np.linalg.inv(P_pred) + H.T @ np.linalg.inv(R) @ H
        )

        x_hat = F @ x_hat + P_filt @ H.T @ np.linalg.inv(R) @ (
            np.array([[medicoes[k].item()]]) - H @ F @ x_hat
        )

        P = P_filt

        estimativas_filtradas.append(x_hat.flatten())
        covariancias_filtradas.append(P.copy())

    return np.array(estimativas_filtradas), np.array(covariancias_filtradas)