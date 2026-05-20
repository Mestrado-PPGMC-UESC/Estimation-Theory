import numpy as np

from model import ModeloPropagacaoInformacao
from simulator import Simulador


class EstimadorLevenbergMarquardt:

    def __init__(self, numero_passos, I0, S0, R0, dados_observados, max_iter=5000, tolerancia=1e-8, epsilon=1e-6, lamb=1e-3):

        # Número de passos da simulação
        self.numero_passos = numero_passos

        # Condições iniciais do sistema
        self.I0 = I0
        self.S0 = S0
        self.R0 = R0

        # Dados reais usados na estimação
        self.dados_observados = dados_observados

        # Configurações do método iterativo
        self.max_iter = max_iter
        self.tolerancia = tolerancia
        self.epsilon = epsilon

        # Parâmetro de amortecimento do Levenberg-Marquardt
        self.lamb = lamb

    def estimar(self, parametros_iniciais):

        # Converte os chutes iniciais em vetor numérico
        parametros = np.array(parametros_iniciais, dtype=float)

        # Históricos para análise da convergência
        historico_beta = []
        historico_alpha = []
        historico_k = []
        historico_erro = []

        # Inicializa o amortecimento
        mu = self.lamb

        # Loop principal do algoritmo
        for iteracao in range(self.max_iter):

            # Calcula o vetor de resíduos
            residuo = self.calcular_residuo(parametros)

            # Calcula a matriz Jacobiana numérica
            J = self.calcular_jacobiana_numerica(parametros, residuo_base=residuo)

            # Calcula erro global atual
            erro = np.linalg.norm(residuo)

            # Salva históricos para análise posterior
            historico_beta.append(parametros[0])
            historico_alpha.append(parametros[1])
            historico_k.append(parametros[2])
            historico_erro.append(erro)

            # Monta o sistema amortecido do Levenberg-Marquardt:
            # (JᵀJ + μI)Δ = Jᵀr
            A = J.T @ J + mu * np.eye(len(parametros))
            b = J.T @ residuo

            # Resolve o sistema para encontrar a correção dos parâmetros
            delta = np.linalg.lstsq(A, b, rcond=None)[0]

            # Atualiza parâmetros com proteção numérica
            parametros, parar = self.atualizar_parametros_com_protecao(parametros, delta)

            # Para se convergiu ou se houve instabilidade numérica
            if parar:
                break

        # Retorna parâmetros estimados, históricos e número de iterações
        return (parametros,historico_beta,historico_alpha,historico_k,historico_erro,iteracao + 1)


    def calcular_residuo(self, parametros):

        # Simula o sistema com os parâmetros atuais
        simulacao = self.simular_com_parametros(parametros)

        # Calcula diferença entre dados reais e simulados
        residuo = self.dados_observados - simulacao

        # Transforma matriz em vetor
        return residuo.flatten()

    def simular_com_parametros(self, parametros):

        # Separa os parâmetros atuais
        beta, alpha, k = parametros

        # Cria o modelo dinâmico
        modelo = ModeloPropagacaoInformacao(beta=beta, alpha=alpha, k=k)

        # Cria o simulador temporal
        simulador = Simulador(modelo=modelo, numero_passos=self.numero_passos)

        # Executa a simulação
        I, S, R = simulador.executar(I0=self.I0, S0=self.S0, R0=self.R0)

        # Organiza os estados em uma matriz
        return np.column_stack((I, S, R))

    def calcular_jacobiana_numerica(self, parametros, residuo_base):

        # Cria matriz Jacobiana:
        # linhas = resíduos
        # colunas = parâmetros
        J = np.zeros((len(residuo_base), len(parametros)))

        # Calcula uma coluna por vez
        for j in range(len(parametros)):

            # Copia parâmetros atuais
            parametros_perturbados = parametros.copy()

            # Perturba apenas um parâmetro
            parametros_perturbados[j] += self.epsilon

            # Calcula novo vetor de resíduos
            residuo_perturbado = self.calcular_residuo(parametros_perturbados)

            # Aproxima derivada numérica por diferenças finitas
            J[:, j] = (residuo_perturbado - residuo_base) / self.epsilon

        return J

    def atualizar_parametros_com_protecao(self, parametros, delta):

        # Atualização do Levenberg-Marquardt:
        # θ_novo = θ - Δ
        parametros_novos = parametros - delta

        # Impede parâmetros negativos ou nulos
        parametros_novos = np.maximum(parametros_novos, 1e-8)

        # Verifica se surgiram nan ou inf
        if not np.all(np.isfinite(parametros_novos)):
            return parametros, True

        # Verifica convergência
        if np.linalg.norm(delta) < self.tolerancia:
            return parametros_novos, True

        # Continua normalmente
        return parametros_novos, False