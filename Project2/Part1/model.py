import numpy as np


class ModeloPreco:

    def __init__(self, taxa_crescimento, volatilidade, passo_tempo, ruido_observacao, tendencia_sistematica):
        self.taxa_crescimento = taxa_crescimento
        self.volatilidade = volatilidade
        self.passo_tempo = passo_tempo
        self.ruido_observacao = ruido_observacao
        self.tendencia_sistematica = tendencia_sistematica

    def proximo_estado(self, estado_atual):

        ruido_processo = np.random.randn()

        return (estado_atual+ (self.taxa_crescimento - 0.5 * self.volatilidade**2) * self.passo_tempo + self.volatilidade * np.sqrt(self.passo_tempo) * ruido_processo)

    def proximo_estado_sem_ruido(self, estado_atual):

        return (estado_atual + (self.taxa_crescimento - 0.5 * self.volatilidade**2) * self.passo_tempo)

    def observacao(self, estado_real):

        ruido_medicao = np.random.randn()

        return (
            estado_real
            + self.tendencia_sistematica
            + self.ruido_observacao * ruido_medicao
        )