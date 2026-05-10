import numpy as np

# -----------------------------
# Dados físicos do problema
# -----------------------------

resistencias = np.array([1.0, 2.0, 3.0])
tensoes_medidas = np.array([2.10, 4.30, 5.85])
correntes_medidas = np.array([2.05, 2.12, 1.90])
confianca_tensao = np.array([5, 4, 3])
confianca_corrente = np.array([4, 4, 2])


# -----------------------------
# Conversão das correntes para tensões equivalentes
# -----------------------------
tensoes_equivalentes = resistencias * correntes_medidas


y = np.concatenate((
    tensoes_medidas,
    tensoes_equivalentes
))


H_tensao = np.diag(resistencias)

H_corrente_convertida = np.diag(resistencias)

H = np.vstack((
    H_tensao,
    H_corrente_convertida
))

confianca = np.concatenate((
    confianca_tensao,
    confianca_corrente
))

W = np.diag(confianca)


G = np.ones((1, len(resistencias)))

u = np.array([6.0])


mus = [0.1, 1, 10, 100, 1000]

for mu in mus:

    x_hat = np.linalg.inv(H.T @ W @ H + mu * (G.T @ G)) @ (H.T @ W @ y + mu * (G.T @ u))

    erro_restricao = abs(np.sum(x_hat) - u[0])

    print(f"\nmu = {mu}")
    print(f"x1 = {x_hat[0]:.4f} A")
    print(f"x2 = {x_hat[1]:.4f} A")
    print(f"x3 = {x_hat[2]:.4f} A")
    print(f"Soma das correntes = {np.sum(x_hat):.4f} A")
    print(f"Erro da restrição = {erro_restricao:.6f}")