import numpy as np


class Simulador:

    def __init__(self, modelo, numero_passos):

        # Modelo dinâmico que define a evolução do sistema
        self.modelo = modelo

        # Quantidade de passos da simulação
        self.numero_passos = numero_passos

    def executar(self, I0, S0, R0):

        # Cria vetores para armazenar toda a evolução temporal
        I = np.zeros(self.numero_passos + 1)
        S = np.zeros(self.numero_passos + 1)
        R = np.zeros(self.numero_passos + 1)

        # Define as condições iniciais
        I[0] = I0
        S[0] = S0
        R[0] = R0

        # Simula a dinâmica do sistema ao longo do tempo
        for i in range(self.numero_passos):

            # Calcula o próximo estado usando o modelo
            I[i + 1], S[i + 1], R[i + 1] = self.modelo.passo(I[i], S[i], R[i])

        # Retorna as trajetórias completas
        return I, S, R