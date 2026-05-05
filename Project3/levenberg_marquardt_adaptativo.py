import numpy as np

from model import ModeloPropagacaoInformacao
from simulator import Simulador


class EstimadorLevenbergMarquardtAdaptativo:

    def __init__(self, numero_passos, I0, S0, R0, dados_observados, max_iter=5000, tolerancia=1e-8, epsilon=1e-6, mu=1e-3):

        # Configurações da simulação
        self.numero_passos = numero_passos
        self.I0 = I0
        self.S0 = S0
        self.R0 = R0

        # Dados reais/observados usados na estimação
        self.dados_observados = dados_observados

        # Configurações do método iterativo
        self.max_iter = max_iter
        self.tolerancia = tolerancia
        self.epsilon = epsilon

        # Valor inicial do amortecimento
        self.mu = mu

    def simular_com_parametros(self, parametros):

        # Separa os parâmetros atuais
        beta, alpha, k = parametros

        # Cria o modelo dinâmico
        modelo = ModeloPropagacaoInformacao(beta=beta, alpha=alpha, k=k)

        # Cria o simulador temporal
        simulador = Simulador(modelo=modelo, numero_passos=self.numero_passos)

        # Executa a simulação
        I, S, R = simulador.executar(I0=self.I0, S0=self.S0, R0=self.R0)

        # Organiza as trajetórias em uma matriz
        return np.column_stack((I, S, R))

    def calcular_residuo(self, parametros):

        # Simula o sistema com os parâmetros atuais
        simulacao = self.simular_com_parametros(parametros)

        # Calcula o erro entre dados reais e simulados
        residuo = self.dados_observados - simulacao

        # Transforma matriz em vetor
        return residuo.flatten()

    def calcular_jacobiana_numerica(self, parametros, residuo_base):

        # Cria matriz Jacobiana: linhas = resíduos, colunas = parâmetros
        J = np.zeros((len(residuo_base), len(parametros)))

        # Calcula uma coluna por vez
        for j in range(len(parametros)):

            # Copia os parâmetros atuais
            parametros_perturbados = parametros.copy()

            # Perturba apenas um parâmetro
            parametros_perturbados[j] += self.epsilon

            # Calcula novo vetor de resíduos
            residuo_perturbado = self.calcular_residuo(parametros_perturbados)

            # Aproxima derivada numérica por diferenças finitas
            J[:, j] = (residuo_perturbado - residuo_base) / self.epsilon

        return J

    def calcular_xi(self, reducao_real, reducao_prevista):

        # Evita divisão por valores muito próximos de zero
        if abs(reducao_prevista) < 1e-15:
            return 0

        # Mede a qualidade do passo dado
        return reducao_real / reducao_prevista

    def aceitar_passo(self, xi):

        # Aceita apenas passos que reduziram o erro
        return xi > 0

    def atualizar_mu(self, xi, mu):

        # Passo ruim: aumenta amortecimento
        if xi < 0.25:
            return 2 * mu

        # Passo bom: aproxima do Gauss-Newton
        if xi > 0.75:
            return mu / 3

        # Passo intermediário: mantém o valor
        return mu

    def calcular_parametros_teste(self, parametros, delta):

        # Calcula parâmetros candidatos
        parametros_teste = parametros - delta

        # Impede parâmetros negativos ou nulos
        parametros_teste = np.maximum(parametros_teste, 1e-8)

        return parametros_teste

    def atualizar_parametros_com_protecao(self, parametros, parametros_teste, delta, xi):

        # Verifica se surgiram nan ou inf
        if not np.all(np.isfinite(parametros_teste)):
            return parametros, True

        # Aceita o passo apenas se ele reduziu o erro
        if self.aceitar_passo(xi):
            parametros = parametros_teste

        # Verifica convergência
        if np.linalg.norm(delta) < self.tolerancia:
            return parametros, True

        # Continua normalmente
        return parametros, False

    def estimar(self, parametros_iniciais):

        # Converte os chutes iniciais em vetor numérico
        parametros = np.array(parametros_iniciais, dtype=float)

        # Históricos para análise da convergência
        historico_beta = []
        historico_alpha = []
        historico_k = []
        historico_erro = []
        historico_mu = []

        # Inicializa amortecimento
        mu = self.mu

        # Loop principal do algoritmo
        for iteracao in range(self.max_iter):

            # Calcula resíduo e jacobiana
            residuo = self.calcular_residuo(parametros)
            J = self.calcular_jacobiana_numerica(parametros, residuo_base=residuo)

            # Calcula erro global atual
            erro_atual = np.linalg.norm(residuo)

            # Salva histórico da iteração
            historico_beta.append(parametros[0])
            historico_alpha.append(parametros[1])
            historico_k.append(parametros[2])
            historico_erro.append(erro_atual)
            historico_mu.append(mu)

            # Monta o sistema amortecido: (JᵀJ + μI)Δ = Jᵀr
            A = J.T @ J + mu * np.eye(len(parametros))
            b = J.T @ residuo

            # Resolve o sistema linear
            delta = np.linalg.lstsq(A, b, rcond=None)[0]

            # Calcula parâmetros candidatos com proteção de positividade
            parametros_teste = self.calcular_parametros_teste(parametros, delta)

            # Calcula erro com os parâmetros candidatos
            residuo_teste = self.calcular_residuo(parametros_teste)
            erro_teste = np.linalg.norm(residuo_teste)

            # Calcula reduções real e prevista
            reducao_real = erro_atual - erro_teste
            reducao_prevista = delta.T @ b

            # Mede a qualidade do passo
            xi = self.calcular_xi(reducao_real, reducao_prevista)

            # Atualiza parâmetros com proteção e checagem do passo
            parametros, parar = self.atualizar_parametros_com_protecao(parametros, parametros_teste, delta, xi)

            # Atualiza amortecimento
            mu = self.atualizar_mu(xi, mu)

            # Para se convergiu ou se houve problema numérico
            if parar:
                break

        # Retorna parâmetros estimados, históricos e iterações
        return (parametros, historico_beta, historico_alpha, historico_k, historico_erro, historico_mu, iteracao + 1)