import numpy as np

from model import ModeloPropagacaoInformacao
from simulator import Simulador


class EstimadorLevenbergMarquardtAdaptativo:

    def __init__(self, numero_passos, I0, S0, R0, dados_observados,
                 max_iter=5000, tolerancia=1e-8, epsilon=1e-6, mu=1e-3):

        self.numero_passos = numero_passos
        self.I0 = I0
        self.S0 = S0
        self.R0 = R0

        self.dados_observados = dados_observados

        self.max_iter = max_iter
        self.tolerancia = tolerancia
        self.epsilon = epsilon

        # Valor inicial do amortecimento
        self.mu = mu

    def simular_com_parametros(self, parametros):

        beta, alpha, k = parametros

        modelo = ModeloPropagacaoInformacao(beta=beta, alpha=alpha, k=k)
        simulador = Simulador(modelo=modelo, numero_passos=self.numero_passos)

        I, S, R = simulador.executar(I0=self.I0, S0=self.S0, R0=self.R0)

        return np.column_stack((I, S, R))

    def calcular_residuo(self, parametros):

        simulacao = self.simular_com_parametros(parametros)
        residuo = self.dados_observados - simulacao

        return residuo.flatten()

    def calcular_jacobiana_numerica(self, parametros, residuo_base):

        J = np.zeros((len(residuo_base), len(parametros)))

        for j in range(len(parametros)):

            parametros_perturbados = parametros.copy()
            parametros_perturbados[j] += self.epsilon

            residuo_perturbado = self.calcular_residuo(parametros_perturbados)

            J[:, j] = (residuo_perturbado - residuo_base) / self.epsilon

        return J

    def calcular_xi(self, reducao_real, reducao_prevista):

        if abs(reducao_prevista) < 1e-15:
            return 0

        return reducao_real / reducao_prevista

    def aceitar_passo(self, xi):

        return xi > 0

    def atualizar_mu(self, xi, mu):

        # Se mu = 0, mantém Gauss-Newton puro
        if mu == 0:
            return 0

        # Passo ruim: aumenta amortecimento
        if xi < 0.25:
            return 2 * mu

        # Passo bom: aproxima do Gauss-Newton
        if xi > 0.75:
            return mu / 3

        # Passo intermediário: mantém o valor
        return mu

    def calcular_parametros_teste(self, parametros, delta):

        parametros_teste = parametros - delta

        # Impede parâmetros negativos ou nulos
        parametros_teste = np.maximum(parametros_teste, 1e-8)

        return parametros_teste

    def atualizar_parametros_com_protecao(self, parametros, parametros_teste, delta, xi):

        if not np.all(np.isfinite(parametros_teste)):
            return parametros, True

        # No modo adaptativo, aceita apenas se reduziu o erro
        if self.aceitar_passo(xi):
            parametros = parametros_teste

        if np.linalg.norm(delta) < self.tolerancia:
            return parametros, True

        return parametros, False

    def atualizar_parametros_sem_xi(self, parametros, parametros_teste, delta):

        if not np.all(np.isfinite(parametros_teste)):
            return parametros, True

        # No caso mu = 0, aceita o passo diretamente como Gauss-Newton
        parametros = parametros_teste

        if np.linalg.norm(delta) < self.tolerancia:
            return parametros, True

        return parametros, False

    def estimar(self, parametros_iniciais):

        parametros = np.array(parametros_iniciais, dtype=float)

        historico_beta = []
        historico_alpha = []
        historico_k = []
        historico_erro = []
        historico_mu = []

        mu = self.mu

        for iteracao in range(self.max_iter):

            residuo = self.calcular_residuo(parametros)
            J = self.calcular_jacobiana_numerica(parametros, residuo_base=residuo)

            erro_atual = np.linalg.norm(residuo)

            historico_beta.append(parametros[0])
            historico_alpha.append(parametros[1])
            historico_k.append(parametros[2])
            historico_erro.append(erro_atual)
            historico_mu.append(mu)

            # Sistema: (JᵀJ + μI)Δ = Jᵀr
            A = J.T @ J + mu * np.eye(len(parametros))
            b = J.T @ residuo

            delta = np.linalg.lstsq(A, b, rcond=None)[0]

            parametros_teste = self.calcular_parametros_teste(parametros, delta)

            residuo_teste = self.calcular_residuo(parametros_teste)
            erro_teste = np.linalg.norm(residuo_teste)

            reducao_real = erro_atual - erro_teste
            reducao_prevista = delta.T @ b

            xi = self.calcular_xi(reducao_real, reducao_prevista)

            # Caso mu = 0: comporta-se como Gauss-Newton puro
            if mu == 0:
                parametros, parar = self.atualizar_parametros_sem_xi(
                    parametros, parametros_teste, delta
                )
            else:
                parametros, parar = self.atualizar_parametros_com_protecao(
                    parametros, parametros_teste, delta, xi
                )

            mu = self.atualizar_mu(xi, mu)

            if parar:
                break

        return (
            parametros,
            historico_beta,
            historico_alpha,
            historico_k,
            historico_erro,
            historico_mu,
            iteracao + 1
        )