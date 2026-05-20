import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

numero_passos = 300
t = np.arange(numero_passos)

# -----------------------------
# Tensão física real
# senoide + ruído de processo
# -----------------------------
sigma_processo = 0.10

tensao_ideal = np.sqrt(2) * 127 * np.sin(0.08 * t)

tensao_real = tensao_ideal + np.random.normal(0.0, sigma_processo, numero_passos)

# -----------------------------
# Sensores reais
# O filtro não conhece esses ruídos
# -----------------------------
sensor1 = tensao_real + np.random.normal(0.0, 5.0, numero_passos)
sensor2 = tensao_real + np.random.normal(0.0, 15.0, numero_passos)
sensor3 = tensao_real + np.random.normal(0.0, 30.0, numero_passos)

medicoes = np.column_stack([sensor1, sensor2, sensor3])

# -----------------------------
# Modelo do filtro
# Estado: [tensao, taxa de variação]
# -----------------------------
dt = 1.0

F = np.array([
    [1.0, dt],
    [0.0, 1.0]
])

Q = np.array([
    [0.05, 0.00],
    [0.00, 0.01]
])

# -----------------------------
# Modelo de medição clássico
# 3 sensores medindo apenas a tensão
# -----------------------------
H_barra = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0]
])

# -----------------------------
# Pesos WA
# w1 + w2 + w3 = 1
# -----------------------------
w1 = 0.70
w2 = 0.20
w3 = 0.10

W_agregacao = np.diag([
    w1,
    w2,
    w3
])

# Nova matriz de medição ponderada
H_WA = W_agregacao @ H_barra

# -----------------------------
# Chute inicial do engenheiro
# R continua representando o ruído assumido dos sensores
# -----------------------------
R1 = 25.0
R2 = 225.0
R3 = 900.0

R_barra = np.diag([
    R1,
    R2,
    R3
])

# -----------------------------
# Inicialização usando a formulação WA
# -----------------------------
P0 = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

z0 = medicoes[0].reshape(3, 1)
z0_WA = W_agregacao @ z0

P = np.linalg.inv(
    np.linalg.inv(P0) + H_WA.T @ np.linalg.inv(R_barra) @ H_WA
)

x_hat = P @ H_WA.T @ np.linalg.inv(R_barra) @ z0_WA
x_hat = x_hat.flatten()

estimativas_tensao = []
estimativas_variacao = []
inovacoes = []

# -----------------------------
# Filtro de Kalman com fusão WA
# -----------------------------
for k in range(numero_passos):

    z = medicoes[k].reshape(3, 1)

    # Aplica WA nas medições
    z_WA = W_agregacao @ z

    # Predição
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q

    # Inovação ponderada
    nu = z_WA.flatten() - H_WA @ x_pred

    inovacoes.append(nu.copy())

    # Ganho de Kalman WA
    S = H_WA @ P_pred @ H_WA.T + R_barra
    K = P_pred @ H_WA.T @ np.linalg.inv(S)

    # Atualização
    x_hat = x_pred + K @ nu
    P = P_pred - K @ H_WA @ P_pred

    estimativas_tensao.append(x_hat[0])
    estimativas_variacao.append(x_hat[1])

estimativas_tensao = np.array(estimativas_tensao)
estimativas_variacao = np.array(estimativas_variacao)
inovacoes = np.array(inovacoes)

# -----------------------------
# Sensor 1
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(sensor1, ".", alpha=0.5, label="Sensor 1")
plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")
plt.title("Sensor 1")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Sensor 2
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(sensor2, ".", alpha=0.5, label="Sensor 2")
plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")
plt.title("Sensor 2")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Sensor 3
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(sensor3, ".", alpha=0.5, label="Sensor 3")
plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")
plt.title("Sensor 3")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# RMSE da estimativa WA
# comparado com sistema ideal
# -----------------------------
rmse = np.sqrt(
    np.mean(
        (tensao_ideal - estimativas_tensao)**2
    )
)

print(f"RMSE da estimativa WA: {rmse:.4f} V")

# -----------------------------
# Validação:
# ideal vs estimativa WA
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(
    tensao_ideal,
    linewidth=2,
    label="Sistema ideal"
)

plt.plot(
    estimativas_tensao,
    linewidth=2,
    label="Estimativa Kalman WA"
)

plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")

plt.title(
    f"Sistema Ideal vs Estimativa WA (RMSE = {rmse:.2f} V)"
)

plt.legend()
plt.grid()
plt.show()