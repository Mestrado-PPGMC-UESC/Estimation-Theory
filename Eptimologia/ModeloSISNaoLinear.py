import numpy as np

from ModeloSIS import ModeloSIS



class ModeloSISNaoLinear(ModeloSIS):

    def f_s(self, S, I):

        return (
            -self.alpha*S*I
            + self.beta*I
            + self.gamma*self.N
            - self.gamma*S
        )


    def f_i(self, S, I):

        return (
            self.alpha*S*I
            - self.beta*I
            - self.gamma*I
        )


    def simular(self):

        t = np.arange(
            0,
            self.tempo_total + self.h,
            self.h
        )

        S = np.zeros(len(t))
        I = np.zeros(len(t))

        S[0] = self.S0
        I[0] = self.I0

        for k in range(len(t)-1):

            S[k+1] = (
                S[k]
                + self.h*self.f_s(
                    S[k],
                    I[k]
                )
            )

            I[k+1] = (
                I[k]
                + self.h*self.f_i(
                    S[k],
                    I[k]
                )
            )

        return t, S, I
    

    def derivadas(self, S, I):
        dS = -self.alpha*S*I + self.beta*I + self.gamma*self.N - self.gamma*S
        dI = self.alpha*S*I - self.beta*I - self.gamma*I
        return dS, dI