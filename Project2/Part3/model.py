import numpy as np


class ModeloPopulacional:

    def __init__(
        self,
        taxa_natalidade,
        taxa_mortalidade,
        ruido_processo,
        ruido_observacao=None,
        ruidos_observacao=None
    ):
        """
        Pode usar:
        - ruido_observacao (1 sensor)
        - ruidos_observacao (lista para múltiplos sensores)
        """

        self.taxa_natalidade = taxa_natalidade
        self.taxa_mortalidade = taxa_mortalidade
        self.ruido_processo = ruido_processo

        # caso 1 sensor
        self.ruido_observacao = ruido_observacao

        # caso multi-sensor
        self.ruidos_observacao = ruidos_observacao

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
        Medição:
        - 1 sensor → escalar
        - multi-sensor → vetor coluna
        """

        # ============================================================
        # MULTI-SENSOR (3 sensores, por exemplo)
        # ============================================================
        if self.ruidos_observacao is not None:
            medidas = []

            for ruido_obs in self.ruidos_observacao:
                ruido = np.random.randn()
                medidas.append(estado_real + ruido_obs * ruido)

            return np.array(medidas).reshape(-1, 1)

        # ============================================================
        # SENSOR ÚNICO (comportamento antigo)
        # ============================================================
        ruido = np.random.randn()
        return estado_real + self.ruido_observacao * ruido