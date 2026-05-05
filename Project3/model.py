class ModeloPropagacaoInformacao:

    def __init__(self, beta, alpha, k):

        # Taxa de propagação da informação
        self.beta = beta

        # Taxa de refutação/esquecimento
        self.alpha = alpha

        # Intensidade de interação entre indivíduos
        self.k = k

    def passo(self, I, S, R):

        # Ignorantes diminuem ao interagir com espalhadores
        novo_I = I - self.beta * self.k * S * I

        # Espalhadores aumentam ao convencer ignorantes,
        # mas diminuem ao serem refutados
        novo_S = S + self.beta * self.k * S * I - self.alpha * S * (S + R)

        # Refutadores aumentam a partir das interações de refutação
        novo_R = R + self.alpha * S * (S + R)

        # Retorna o próximo estado do sistema
        return novo_I, novo_S, novo_R