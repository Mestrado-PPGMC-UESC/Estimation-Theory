import numpy as np


def obter_parametros(tempo, q_valor=0.01, r_valor=0.5):

    dt = np.mean(np.diff(tempo))

    F = np.array([
        [1.0, dt, 0.5 * dt**2],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0]
    ])

    H = np.array([
        [0.0, 0.0, 1.0]
    ])

    Q = np.array([
        [q_valor, 0.0, 0.0],
        [0.0, q_valor, 0.0],
        [0.0, 0.0, q_valor]
    ])

    R = np.array([
        [r_valor]
    ])

    return dt, F, H, Q, R