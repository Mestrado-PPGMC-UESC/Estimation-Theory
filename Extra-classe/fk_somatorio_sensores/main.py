import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

numero_passos = 300
t = np.arange(numero_passos)

# -----------------------------
# tensao física real
# senoide + ruído de processo
# -----------------------------
sigma_processo = 0.10

tensao_ideal = np.sqrt(2)*127*np.sin(0.08*t)

tensao_real = tensao_ideal + np.random.normal(0.0, sigma_processo, numero_passos)

# -----------------------------
# Sensores reais
# O filtro não conhece esses ruídos
# -----------------------------

# Sensores reais
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

# 3 sensores medindo apenas a tensao
H_barra = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0]
])

# -----------------------------
# Chute inicial do engenheiro
# tuning manual dos sensores
# -----------------------------
R1 = 1.0
R2 = 5.0
R3 = 10.0

R_barra = np.diag([R1, R2, R3])

# -----------------------------
# Inicialização
# -----------------------------
x_hat = np.array([
    0.0,
    0.0
])

P = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

estimativas_tensao = []
estimativas_variacao = []
inovacoes = []

# -----------------------------
# Filtro de Kalman com fusão
# -----------------------------
for k in range(numero_passos):

    z = medicoes[k].reshape(3, 1)

    # Predição
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q

    # Inovação
    nu = z.flatten() - H_barra @ x_pred

    inovacoes.append(nu.copy())

    # Ganho de Kalman
    S = H_barra @ P_pred @ H_barra.T + R_barra
    K = P_pred @ H_barra.T @ np.linalg.inv(S)

    # Atualização
    x_hat = x_pred + K @ nu
    P = P_pred - K @ H_barra @ P_pred

    estimativas_tensao.append(x_hat[0])
    estimativas_variacao.append(x_hat[1])

estimativas_tensao = np.array(estimativas_tensao)
estimativas_variacao = np.array(estimativas_variacao)
inovacoes = np.array(inovacoes)

# -----------------------------
# Sensor 1 vs Estimativa
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(
    sensor1,
    ".",
    alpha=0.5,
    label="Sensor 1"
)


plt.xlabel("Tempo")
plt.ylabel("tensao (V)")
plt.title("Sensor 1 ")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# Sensor 2 vs Estimativa
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(
    sensor2,
    ".",
    alpha=0.5,
    label="Sensor 2"
)

plt.xlabel("Tempo")
plt.ylabel("tensao (v)")
plt.title("Sensor 2 ")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# Sensor 3 vs Estimativa
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(
    sensor3,
    ".",
    alpha=0.5,
    label="Sensor 3"
)

plt.xlabel("Tempo")
plt.ylabel("tensao (A)")
plt.title("Sensor 3 ")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Estimativa final da fusão
# -----------------------------
# -----------------------------
# RMSE da estimativa
# comparado com sistema ideal
# -----------------------------
rmse = np.sqrt(
    np.mean(
        (tensao_ideal - estimativas_tensao)**2
    )
)

print(f"RMSE da estimativa: {rmse:.4f} V")

# -----------------------------
# Validação:
# ideal vs estimativa
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
    label="Estimativa Kalman"
)

plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")

plt.title(
    f"Sistema Ideal vs Estimativa (RMSE = {rmse:.2f} V)"
)

plt.legend()
plt.grid()
plt.show()