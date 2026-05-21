import numpy as np
from ModeloSIS import ModeloSIS


class ModeloSISPontoEstavel(ModeloSIS):

    def __init__(self, alpha, beta, gamma, N, S0, I0, passo_tempo, tempo_total):
        super().__init__(alpha, beta, gamma, N, S0, I0, passo_tempo, tempo_total)

        self.S_eq = (self.beta + self.gamma) / self.alpha
        self.I_eq = self.N - self.S_eq

    def derivadas(self, S, I):
        u = S - self.S_eq
        v = I - self.I_eq

        du = (-self.alpha*self.I_eq - self.gamma)*u + (-self.alpha*self.S_eq + self.beta)*v
        dv = (self.alpha*self.I_eq)*u + (self.alpha*self.S_eq - self.beta - self.gamma)*v

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