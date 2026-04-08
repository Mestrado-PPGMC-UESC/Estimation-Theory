import numpy as np


def estimar_parametros(S, I, R):
    
    # ESTIMAÇÃO BETA

    # y_beta = S_k - S_{k+1}
    
    y_beta = S[:-1] - S[1:]

    # H_beta = S_k * I_k
    h_beta = S[:-1] * I[:-1]

    numerador_beta = np.sum(h_beta * y_beta)
    denominador_beta = np.sum(h_beta**2)

    beta_estimado = numerador_beta / denominador_beta

    # ESTIMAÇÃO GAMMA


    y_gamma = R[1:] - R[:-1]

    # H_gamma = I_k
    h_gamma = I[:-1]

    numerador_gamma = np.sum(h_gamma * y_gamma)
    denominador_gamma = np.sum(h_gamma**2)

    gamma_estimado = numerador_gamma / denominador_gamma


    return beta_estimado,gamma_estimado