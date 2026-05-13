import numpy as np


def simular_sistema(F, H, Q, R, x_inicial, numero_passos):

    x_real = x_inicial.copy()

    x_perfeito = x_inicial.copy()

    estados_reais = []

    estados_perfeitos = []

    medicoes = []

    for k in range(numero_passos):

        # Trajetória perfeita (sem ruído)
        x_perfeito = F @ x_perfeito

        # Ruídos
        w = np.random.multivariate_normal(
            mean=np.zeros(len(x_inicial)),
            cov=Q
        )

        v = np.random.normal(
            loc=0.0,
            scale=np.sqrt(R[0, 0])
        )

        # Trajetória real (com ruído)
        x_real = F @ x_real + w

        # Medição
        y = H @ x_real + v

        estados_perfeitos.append(
            x_perfeito.copy()
        )

        estados_reais.append(
            x_real.copy()
        )

        medicoes.append(
            y[0]
        )

    return (
        np.array(estados_reais),
        np.array(estados_perfeitos),
        np.array(medicoes)
    )