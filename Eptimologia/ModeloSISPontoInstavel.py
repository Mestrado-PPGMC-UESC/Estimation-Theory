import numpy as np
from ModeloSIS import ModeloSIS


class ModeloSISPontoInstavel(ModeloSIS):

    def __init__(self, alpha, beta, gamma, N, S0, I0, passo_tempo, tempo_total):
        super().__init__(alpha, beta, gamma, N, S0, I0, passo_tempo, tempo_total)

        self.S_eq = self.N
        self.I_eq = 0

    def derivadas(self, S, I):
        u = S - self.S_eq
        v = I - self.I_eq

        du = (-self.gamma)*u + (-self.alpha*self.N + self.beta)*v
        dv = 0*u + (self.alpha*self.N - self.beta - self.gamma)*v

        return du, dv

    def simular(self):
        t = np.arange(0, self.tempo_total + self.h, self.h)

        S = np.zeros(len(t))
        I = np.zeros(len(t))

        S[0] = self.S0
        I[0] = self.I0

        for k in range(len(t) - 1):
            dS, dI = self.derivadas(S[k], I[k])

            S[k + 1] = S[k] + self.h*dS
            I[k + 1] = I[k] + self.h*dI

        return t, S, I