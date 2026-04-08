import numpy as np

def simular_sir(beta,gamma,S0,I0,R0, n_passos):
    S = np.zeros(n_passos + 1)
    I = np.zeros(n_passos + 1)
    R = np.zeros(n_passos + 1)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for k in range(n_passos):
        S[k+1] = S[k] - beta*S[k]*I[k]
        I[k+1] = I[k] + beta*S[k]*I[k] - gamma*I[k]
        R[k+1] = R[k] + gamma*I[k]
    
    return S,I,R