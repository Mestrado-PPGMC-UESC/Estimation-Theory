import numpy as np

class FiltroKalmanExtendido:

    def __init__(self,modelo,Q,R,P0):

        self.modelo = modelo
        self.Q = Q
        self.R = R
        self.P = P0

    def calcular_estado_predito(self,estado_estimado):
        
        I = estado_estimado[0]
        S = estado_estimado[1]
        R = estado_estimado[2]

        novo_I,novo_S,novo_R = self.modelo.passo(I,S,R)

        estado_predito = np.array([novo_I,novo_S,novo_R])

        return estado_predito
    
    def calcular_jacobiana_estado(self,estado_estimado):

        I = estado_estimado[0]
        S = estado_estimado[1]
        R = estado_estimado[2]

        beta = self.modelo.beta
        alpha = self.modelo.alpha
        k = self.modelo.k

        matriz_F = np.array([
            [
                1 - beta * k * S,
                -beta * k * I,
                0
            ],
            [
                beta * k * S,
                1 + beta * k * I - alpha * (2 * S + R),
                -alpha * S
            ],
            [
                0,
                alpha * (2 * S + R),
                1 + alpha * S
            ]
        ])

        return matriz_F
    
    def calcular_jacobiana_observacao(self):
        matriz_H = np.eye(3)
        return matriz_H
    
    def etapa_predicao(self,estado_estimado):

        estado_predito = self.calcular_estado_predito(estado_estimado)
        matriz_F = self.calcular_jacobiana_estado(estado_estimado)
        P_predito = matriz_F @ self.P @ matriz_F.T + self.Q
        return estado_predito,P_predito

    def etapa_correcao(self,estado_predito,P_predito,observacao):

        matriz_H = self.calcular_jacobiana_observacao()
        inovacao = observacao - estado_predito
        matriz_S = matriz_H @ P_predito @ matriz_H.T + self.R
        ganho_kalman = P_predito @ matriz_H.T @ np.linalg.inv(matriz_S)
        estado_corrigido = estado_predito + ganho_kalman @ inovacao
        identidade = np.eye(3)
        P_corrigido = (identidade - ganho_kalman @ matriz_H) @ P_predito

        self.P = P_corrigido
        return estado_corrigido,P_corrigido
    
    def filtrar(self,observacoes,estado_inicial):

        estado_estimado = np.array(estado_inicial,dtype=float)

        historico_estados = []

        for i in range(len(observacoes)):

            observacao_atual = observacoes[i]

            estado_predito, P_predito = self.etapa_predicao(estado_estimado)

            estado_corrigido,P_corrigido = self.etapa_correcao(estado_predito,P_predito,observacao_atual)

            estado_estimado = estado_corrigido

            historico_estados.append(estado_estimado.copy())

        return np.array(historico_estados)