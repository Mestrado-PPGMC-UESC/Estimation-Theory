import numpy as np


def obter_matrizes_kalman(modelo, estado_inicial):
    """
    Monta F, H, Q, R, x0 e P0 de acordo com o tipo de modelo.
    """

    # ============================================================
    # Modelo de Preço
    # ============================================================
    if hasattr(modelo, "taxa_crescimento"):
        mu = modelo.taxa_crescimento
        sigma = modelo.volatilidade
        dt = modelo.passo_tempo
        q = modelo.ruido_observacao
        eta_r = modelo.tendencia_sistematica

        b = (mu - 0.5 * sigma**2) * dt

        F = np.array([
            [1, b],
            [0, 1]
        ])

        H = np.array([[1, eta_r]])

        Q = np.array([
            [sigma**2 * dt, 0],
            [0, 0]
        ])

        R = np.array([[q**2]])

        x0 = np.array([
            [estado_inicial],
            [1]
        ])

        P0 = np.eye(2)

        return F, H, Q, R, x0, P0

    # ============================================================
    # Modelo Populacional
    # ============================================================
    if hasattr(modelo, "taxa_natalidade"):
        alpha = modelo.taxa_natalidade
        beta = modelo.taxa_mortalidade

        F = np.array([[1 + alpha - beta]])
        H = np.array([[1]])

        Q = np.array([[modelo.ruido_processo**2]])
        R = np.array([[modelo.ruido_observacao**2]])

        x0 = np.array([[estado_inicial]])
        P0 = np.eye(1)

        return F, H, Q, R, x0, P0

    raise ValueError("Modelo não reconhecido para montagem das matrizes de Kalman.")


def passo_predicao(F, Q, x_corrigido, P_corrigido):
    x_predito = F @ x_corrigido
    P_predito = F @ P_corrigido @ F.T + Q

    return x_predito, P_predito


def passo_correcao(H, R, observacao, x_predito, P_predito):
    inovacao = observacao - H @ x_predito

    S = H @ P_predito @ H.T + R
    K = P_predito @ H.T @ np.linalg.inv(S)

    x_corrigido = x_predito + K @ inovacao
    P_corrigido = P_predito - K @ H @ P_predito

    return x_corrigido, P_corrigido


def estimativa_filtrada(observacoes, modelo, estado_inicial):
    """
    Calcula a estimativa filtrada x_hat(k|k).
    """

    F, H, Q, R, x_corrigido, P_corrigido = obter_matrizes_kalman(
        modelo,
        estado_inicial
    )

    numero_passos = len(observacoes)
    estimativas_filtradas = np.zeros(numero_passos)

    for k in range(numero_passos):
        x_predito, P_predito = passo_predicao(
            F,
            Q,
            x_corrigido,
            P_corrigido
        )

        x_corrigido, P_corrigido = passo_correcao(
            H,
            R,
            observacoes[k],
            x_predito,
            P_predito
        )

        estimativas_filtradas[k] = x_corrigido[0, 0]

    return estimativas_filtradas


def estimativa_preditiva(observacoes, modelo, estado_inicial):
    """
    Calcula a estimativa preditiva x_hat(k+1|k),
    na forma recursiva do material.
    """

    F, H, Q, R, x_predito, P_predito = obter_matrizes_kalman(
        modelo,
        estado_inicial
    )

    numero_passos = len(observacoes)
    estimativas_preditivas = np.zeros(numero_passos)
    estimativas_preditivas[0] = x_predito[0, 0]

    for k in range(numero_passos - 1):
        S = H @ P_predito @ H.T + R

        ganho_preditivo = F @ P_predito @ H.T @ np.linalg.inv(S)
        inovacao = observacoes[k] - H @ x_predito

        x_predito = F @ x_predito + ganho_preditivo @ inovacao

        P_predito = (
            F @ P_predito @ F.T
            + Q
            - F @ P_predito @ H.T @ np.linalg.inv(S) @ H @ P_predito @ F.T
        )

        estimativas_preditivas[k + 1] = x_predito[0, 0]

    return estimativas_preditivas


def estimativa_preditiva_corretiva(observacoes, modelo, estado_inicial):
    """
    Aplica o Filtro de Kalman no formato preditivo-corretivo.

    Retorna:
        estimativas_preditivas: x_hat(k|k-1)
        estimativas_corrigidas: x_hat(k|k)
    """

    F, H, Q, R, x_corrigido, P_corrigido = obter_matrizes_kalman(
        modelo,
        estado_inicial
    )

    numero_passos = len(observacoes)

    estimativas_preditivas = np.zeros(numero_passos)
    estimativas_corrigidas = np.zeros(numero_passos)

    for k in range(numero_passos):
        x_predito, P_predito = passo_predicao(
            F,
            Q,
            x_corrigido,
            P_corrigido
        )

        estimativas_preditivas[k] = x_predito[0, 0]

        x_corrigido, P_corrigido = passo_correcao(
            H,
            R,
            observacoes[k],
            x_predito,
            P_predito
        )

        estimativas_corrigidas[k] = x_corrigido[0, 0]

    return estimativas_preditivas, estimativas_corrigidas