import numpy as np

import numpy as np


def estimativa_filtrada(observacoes, modelo, estado_inicial=None):
    """
    Calcula a estimativa filtrada do Filtro de Kalman usando exatamente
    a forma recursiva apresentada no material do professor.

    Retorna x_hat(k|k), isto é, a estimativa do estado após incorporar
    a medição do instante k.
    """

    numero_passos = len(observacoes)

    mu = modelo.taxa_crescimento
    sigma = modelo.volatilidade
    dt = modelo.passo_tempo
    q = modelo.ruido_observacao
    eta_r = modelo.tendencia_sistematica

    b = (mu - 0.5 * sigma**2) * dt

    # F_k: matriz de transição de estados
    F = np.array([
        [1, b],
        [0, 1]
    ])

    # H_k: matriz de observação
    H = np.array([[1, eta_r]])

    # Q_k: covariância do ruído de processo
    Q = np.array([
        [sigma**2 * dt, 0],
        [0, 0]
    ])

    # R_k: covariância do ruído de observação
    R = np.array([[q**2]])

    R_inv = np.linalg.inv(R)

    estimativas_filtradas = np.zeros(numero_passos)

    # ============================================================
    # Passo 0: Inicialização
    # P(0|0) = (P0^(-1) + H0^T R0^(-1) H0)^(-1)
    # x_hat(0|0) = P(0|0) H0^T R0^(-1) y0
    # ============================================================

    P0 = np.eye(2)

    y0 = np.array([[observacoes[0]]])

    P = np.linalg.inv(
        np.linalg.inv(P0) + H.T @ R_inv @ H
    )

    x_estimado = P @ H.T @ R_inv @ y0

    estimativas_filtradas[0] = x_estimado[0, 0]

    # ============================================================
    # Passo 1: Atualização Recursiva
    # ============================================================

    for k in range(numero_passos - 1):

        y_proximo = np.array([[observacoes[k + 1]]])

        P_predito = F @ P @ F.T + Q

        P = np.linalg.inv(
            np.linalg.inv(P_predito) + H.T @ R_inv @ H
        )

        inovacao = y_proximo - H @ F @ x_estimado

        x_estimado = (
            F @ x_estimado
            + P @ H.T @ R_inv @ inovacao
        )

        estimativas_filtradas[k + 1] = x_estimado[0, 0]

    return estimativas_filtradas


def estimativa_preditiva(observacoes, modelo, estado_inicial):
    """
    Calcula a estimativa preditiva do Filtro de Kalman na forma recursiva.

    Retorna a sequência x_hat(k|k-1), isto é, a previsão do estado
    antes da incorporação da medição do instante k.
    """

    numero_passos = len(observacoes)

    mu = modelo.taxa_crescimento
    sigma = modelo.volatilidade
    dt = modelo.passo_tempo
    q = modelo.ruido_observacao
    eta_r = modelo.tendencia_sistematica

    b = (mu - 0.5 * sigma**2) * dt

    A = np.array([
        [1, b],
        [0, 1]
    ])

    C = np.array([[1, eta_r]])

    Q = np.array([
        [sigma**2 * dt, 0],
        [0, 0]
    ])

    R = np.array([[q**2]])

    x_predito = np.array([
        [estado_inicial],
        [1]
    ])

    P_predito = np.eye(2)

    estimativas_preditivas = np.zeros(numero_passos)
    estimativas_preditivas[0] = x_predito[0, 0]

    for k in range(numero_passos - 1):

        S = C @ P_predito @ C.T + R

        ganho_preditivo = A @ P_predito @ C.T @ np.linalg.inv(S)

        inovacao = observacoes[k] - C @ x_predito

        x_predito_proximo = A @ x_predito + ganho_preditivo @ inovacao

        P_predito_proximo = (
            A @ P_predito @ A.T
            + Q
            - A @ P_predito @ C.T @ np.linalg.inv(S) @ C @ P_predito @ A.T
        )

        x_predito = x_predito_proximo
        P_predito = P_predito_proximo

        estimativas_preditivas[k + 1] = x_predito[0, 0]

    return estimativas_preditivas



def estimativa_preditiva_corretiva(observacoes, modelo, estado_inicial):
    """
    Aplica o Filtro de Kalman no formato preditivo-corretivo,
    seguindo a notação do material:

        F_k: matriz de transição de estado
        H_k: matriz de observação
        Q_k: covariância do ruído de processo
        R_k: covariância do ruído de medição

    Retorna:
        estimativas_preditivas: x_hat(k|k-1)
        estimativas_corrigidas: x_hat(k|k)
    """

    numero_passos = len(observacoes)

    mu = modelo.taxa_crescimento
    sigma = modelo.volatilidade
    dt = modelo.passo_tempo
    q = modelo.ruido_observacao
    eta_r = modelo.tendencia_sistematica

    b = (mu - 0.5 * sigma**2) * dt

    F_k = np.array([
        [1, b],
        [0, 1]
    ])

    H_k = np.array([[1, eta_r]])

    Q_k = np.array([
        [sigma**2 * dt, 0],
        [0, 0]
    ])

    R_k = np.array([[q**2]])

    x_corrigido = np.array([
        [estado_inicial],
        [1]
    ])

    P_corrigido = np.eye(2)
    I = np.eye(2)

    estimativas_preditivas = np.zeros(numero_passos)
    estimativas_corrigidas = np.zeros(numero_passos)

    for k in range(numero_passos):

        # ============================================================
        # Passo 1: Predição
        # x_hat(k|k-1) = F_k x_hat(k-1|k-1)
        # P(k|k-1) = F_k P(k-1|k-1) F_k^T + Q_k
        # ============================================================
        x_predito = F_k @ x_corrigido
        P_predito = F_k @ P_corrigido @ F_k.T + Q_k

        estimativas_preditivas[k] = x_predito[0, 0]

        # ============================================================
        # Passo 2: Correção
        # K_k = P(k|k-1) H_k^T (H_k P(k|k-1) H_k^T + R_k)^(-1)
        # x_hat(k|k) = x_hat(k|k-1) + K_k (y_k - H_k x_hat(k|k-1))
        # P(k|k) = P(k|k-1) - K_k H_k P(k|k-1)
        # ============================================================
        inovacao = observacoes[k] - H_k @ x_predito

        S = H_k @ P_predito @ H_k.T + R_k
        K_k = P_predito @ H_k.T @ np.linalg.inv(S)

        x_corrigido = x_predito + K_k @ inovacao
        P_corrigido = P_predito - K_k @ H_k @ P_predito

        estimativas_corrigidas[k] = x_corrigido[0, 0]

    return estimativas_preditivas, estimativas_corrigidas