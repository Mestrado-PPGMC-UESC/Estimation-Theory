import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

numero_passos = 300
t = np.arange(numero_passos)

# -----------------------------
# Tensão física real
# -----------------------------
sigma_processo = 0.10

tensao_ideal = np.sqrt(2) * 127 * np.sin(0.08 * t)
tensao_real = tensao_ideal + np.random.normal(0.0, sigma_processo, numero_passos)

# -----------------------------
# Sensores reais
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

H_barra = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0]
])

# -----------------------------
# Pesos OWA
# w1 pesa o maior valor, w2 o intermediário, w3 o menor
# -----------------------------
pesos_owa = np.array([
    0.10,
    0.80,
    0.10
])

W_owa = np.diag(pesos_owa)

# -----------------------------
# R assumido dos sensores
# -----------------------------
R1 = 25.0
R2 = 225.0
R3 = 900.0

R_barra_original = np.diag([
    R1,
    R2,
    R3
])

# -----------------------------
# Inicialização usando OWA
# -----------------------------
P0 = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

z0 = medicoes[0]

# Ordena as medições do maior para o menor
ordem0 = np.argsort(z0)[::-1]

z0_ordenado = z0[ordem0].reshape(3, 1)
H_ordenado = H_barra[ordem0, :]
R_ordenado = R_barra_original[np.ix_(ordem0, ordem0)]

H_OWA = W_owa @ H_ordenado
z0_OWA = W_owa @ z0_ordenado

P = np.linalg.inv(
    np.linalg.inv(P0) + H_OWA.T @ np.linalg.inv(R_ordenado) @ H_OWA
)

x_hat = P @ H_OWA.T @ np.linalg.inv(R_ordenado) @ z0_OWA
x_hat = x_hat.flatten()

estimativas_tensao = []
estimativas_variacao = []
inovacoes = []
ordens = []

# -----------------------------
# Filtro de Kalman com fusão OWA
# -----------------------------
for k in range(numero_passos):

    z_atual = medicoes[k]

    # Ordena as medições do maior para o menor
    ordem = np.argsort(z_atual)[::-1]

    z_ordenado = z_atual[ordem].reshape(3, 1)
    H_ordenado = H_barra[ordem, :]
    R_ordenado = R_barra_original[np.ix_(ordem, ordem)]

    # Aplica OWA
    H_OWA = W_owa @ H_ordenado
    z_OWA = W_owa @ z_ordenado

    # Predição
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q

    # Inovação OWA
    nu = z_OWA.flatten() - H_OWA @ x_pred

    inovacoes.append(nu.copy())
    ordens.append(ordem.copy())

    # Ganho de Kalman OWA
    S = H_OWA @ P_pred @ H_OWA.T + R_ordenado
    K = P_pred @ H_OWA.T @ np.linalg.inv(S)

    # Atualização
    x_hat = x_pred + K @ nu
    P = P_pred - K @ H_OWA @ P_pred

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
# RMSE da estimativa OWA
# -----------------------------
rmse = np.sqrt(
    np.mean(
        (tensao_ideal - estimativas_tensao)**2
    )
)

print(f"RMSE da estimativa OWA: {rmse:.4f} V")

# -----------------------------
# Validação
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
    label="Estimativa Kalman OWA"
)

plt.xlabel("Tempo")
plt.ylabel("Tensão (V)")

plt.title(
    f"Sistema Ideal vs Estimativa OWA (RMSE = {rmse:.2f} V)"
)

plt.legend()
plt.grid()
plt.show()