import numpy as np


class ModeloPopulacional:

    def __init__(self, taxa_natalidade, taxa_mortalidade, ruido_processo, ruido_observacao):
        self.taxa_natalidade = taxa_natalidade
        self.taxa_mortalidade = taxa_mortalidade
        self.ruido_processo = ruido_processo
        self.ruido_observacao = ruido_observacao

    def proximo_estado(self, estado_atual):
        """
        Dinâmica com ruído de processo
        x_{k+1} = (1 + alpha - beta) x_k + w_k
        """

        ruido = np.random.randn()

        F = 1 + self.taxa_natalidade - self.taxa_mortalidade

        return F * estado_atual + self.ruido_processo * ruido

    def proximo_estado_sem_ruido(self, estado_atual):
        """
        Dinâmica determinística
        """

        F = 1 + self.taxa_natalidade - self.taxa_mortalidade

        return F * estado_atual

    def observacao(self, estado_real):
        """
        Medição com ruído
        y_k = x_k + v_k
        """

        ruido = np.random.randn()

        return estado_real + self.ruido_observacao * ruido