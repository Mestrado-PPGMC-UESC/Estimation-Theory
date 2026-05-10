import numpy as np

# -----------------------------
# Dados experimentais
# -----------------------------
correntes_medidas = np.array([2.10, 1.85, 2.35])  # A
tensao_fonte = 12.0                               # V
confianca = np.array([5, 4, 3])

# -----------------------------
# Modelo de medição
# y = Hx + v
# -----------------------------
H = np.diag(tensao_fonte * np.ones(len(correntes_medidas)))
W = np.diag(confianca)

# Tensões medidas pelos sensores (Lei de Ohm)
y = tensao_fonte * correntes_medidas

# -----------------------------
# Restrição física
# x1 + x2 + x3 = 6.25 A
# -----------------------------
G = np.ones((1, len(correntes_medidas)))
u = np.array([6.25])

# Pesos da restrição
mus = [0.1, 1, 10, 100, 1000]

for mu in mus:

    x_hat = np.linalg.inv(H.T @ W @ H + mu * (G.T @ G)) @ (H.T @ W @ y + mu * (G.T @ u))


    # Potências estimadas
    potencias = tensao_fonte * x_hat

    # Erro da restrição
    erro_restricao = abs(np.array([np.sum(x_hat)]) - u)

    print(f"\nmu = {mu}")
    print("-" * 40)
    print(f"I1 = {x_hat[0]:.4f} A   |   P1 = {potencias[0]:.4f} W")
    print(f"I2 = {x_hat[1]:.4f} A   |   P2 = {potencias[1]:.4f} W")
    print(f"I3 = {x_hat[2]:.4f} A   |   P3 = {potencias[2]:.4f} W")
    print(f"Soma das correntes = {np.sum(x_hat):.4f} A")
    print(f"Soma das potências = {np.sum(potencias):.4f} W")
    print(f"Erro da restrição = {erro_restricao[0]:.6f}")