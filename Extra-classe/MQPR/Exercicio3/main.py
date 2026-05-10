import numpy as np

consumo_medido = np.array([3.2, 2.8, 3.5])
confianca = np.array([5, 4, 3])

H = np.eye(len(consumo_medido))
W = np.diag(confianca)
y = consumo_medido

G = np.ones((1, len(consumo_medido)))
u = [10.0]

mus = [0.1, 1, 10, 100, 1000]

for mu in mus:

    x_hat = np.linalg.inv(H.T @ W @ H + mu * (G.T @ G)) @ (H.T @ W @ y + mu * (G.T @ u))


    erro_restricao = abs(np.sum(x_hat) - u)

    print(f"\nmu = {mu}")
    print(f"x1 = {x_hat[0]:.4f} L/h")
    print(f"x2 = {x_hat[1]:.4f} L/h")
    print(f"x3 = {x_hat[2]:.4f} L/h")
    print(f"Soma = {np.sum(x_hat):.4f} L/h")
    print(f"Erro da restrição = {erro_restricao[0]:.6f}")