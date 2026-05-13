import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

dt = 1.0
numero_passos = 20

x_real = np.array([0.0, 1.0])

F = np.array([
    [1.0, dt],
    [0.0, 1.0]
])

H = np.array([
    [1.0, 0.0]
])

Q = np.array([
    [0.01, 0.00],
    [0.00, 0.01]
])

R = np.array([
    [10.0]
])

estados_reais = []
medicoes = []

for k in range(numero_passos):
    w = np.random.multivariate_normal(mean=[0, 0], cov=Q)
    v = np.random.normal(loc=0, scale=np.sqrt(R[0, 0]))
    x_real = F @ x_real + w
    y = H @ x_real + v
    estados_reais.append(x_real.copy())
    medicoes.append(y[0])

estados_reais = np.array(estados_reais)
medicoes = np.array(medicoes)

# -----------------------------
# Estimativa preditivo-corretiva
# -----------------------------

x_hat = np.array([0.0, 0.0])

P = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

estimativas_preditoras = []
estimativas_corrigidas = []

for k in range(numero_passos):

    # Predição
    x_preditor = F @ x_hat
    P_preditor = F @ P @ F.T + Q

    # Correção
    S = H @ P_preditor @ H.T + R
    K = P_preditor @ H.T @ np.linalg.inv(S)

    inovacao = np.array([medicoes[k]]) - H @ x_preditor

    x_corrigido = x_preditor + K @ inovacao
    P_corrigido = (np.eye(2) - K @ H) @ P_preditor

    estimativas_preditoras.append(x_preditor.copy())
    estimativas_corrigidas.append(x_corrigido.copy())

    x_hat = x_corrigido
    P = P_corrigido

estimativas_preditoras = np.array(estimativas_preditoras)
estimativas_corrigidas = np.array(estimativas_corrigidas)

# -----------------------------
# Gráfico da posição
# -----------------------------

plt.figure()
plt.plot(estados_reais[:, 0], label="Posição real")
plt.plot(medicoes, ".", label="Medições")
plt.plot(estimativas_preditoras[:, 0], label="Estimativa preditora")
plt.plot(estimativas_corrigidas[:, 0], label="Estimativa corrigida")
plt.xlabel("Tempo")
plt.ylabel("Posição")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Gráfico da velocidade
# -----------------------------

plt.figure()
plt.plot(estados_reais[:, 1], label="Velocidade real")
plt.plot(estimativas_preditoras[:, 1], label="Velocidade preditora")
plt.plot(estimativas_corrigidas[:, 1], label="Velocidade corrigida")
plt.xlabel("Tempo")
plt.ylabel("Velocidade")
plt.legend()
plt.grid()
plt.show()