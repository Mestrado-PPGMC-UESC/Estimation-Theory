import numpy as np

tensoes_medidas = np.array([2.62,2.74,2.91,3.05,3.24])
confianca = np.array([5,4,3,2,1])

H = np.array([
    [1.0, 0.5, 0.2],
    [0.8, 0.7, 0.3],
    [0.6, 1.0, 0.5],
    [0.4, 1.2, 0.8],
    [0.3, 1.5, 1.0]
])

y = tensoes_medidas
W = np.diag(confianca)
G = np.ones((1, 3))
u = [6.0]

mus = [0.1, 1, 10, 100, 1000]

for mu in mus:

    x_hat = np.linalg.inv(H.T @ W @ H + mu * (G.T @ G)) @ (H.T @ W @ y + mu * (G.T @ u))
    
    erro_restricao = abs(np.sum(x_hat) - u)

    print(f"\nmu = {mu}")
    print(f"x1 = {x_hat[0]:.4f} A")
    print(f"x2 = {x_hat[1]:.4f} A")
    print(f"x3 = {x_hat[2]:.4f} A")
    print(f"Soma = {np.sum(x_hat):.4f} A")
    print(f"Erro da restrição = {erro_restricao[0]:.6f}")