from abc import ABC, abstractmethod

class ModeloSIS(ABC):

    def __init__(
        self,
        alpha,
        beta,
        gamma,
        N,
        S0,
        I0,
        passo_tempo,
        tempo_total
    ):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.N = N

        self.S0 = S0
        self.I0 = I0

        self.h = passo_tempo
        self.tempo_total = tempo_total