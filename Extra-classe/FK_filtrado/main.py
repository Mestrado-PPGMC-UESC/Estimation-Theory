import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

dt = 1.0
numero_passos = 20

# estado: posição / velocidade
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
    [1.0]
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

x_hat = np.array([0.0, 0.0])

P = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

estimativas = []

for k in range(numero_passos):

    P_filt = np.linalg.inv(
        np.linalg.inv(Q + F @ P @ F.T) + H.T @ np.linalg.inv(R) @ H
    )

    x_hat = F @ x_hat + P_filt @ H.T @ np.linalg.inv(R) @ (
        np.array([medicoes[k]]) - H @ F @ x_hat
    )

    P = P_filt

    estimativas.append(x_hat.copy())

estimativas = np.array(estimativas)

# -----------------------------
# Conversão para array
# -----------------------------
estimativas = np.array(estimativas)

# -----------------------------
# Gráfico da posição
# -----------------------------
plt.figure()

plt.plot(
    estados_reais[:, 0],
    label="Posição real"
)

plt.plot(
    medicoes,
    ".",
    label="Medições"
)

plt.plot(
    estimativas[:, 0],
    label="Estimativa Kalman"
)

plt.xlabel("Tempo")
plt.ylabel("Posição")
plt.legend()
plt.grid()

plt.show()


# -----------------------------
# Gráfico da velocidade
# -----------------------------
plt.figure()

plt.plot(
    estados_reais[:, 1],
    label="Velocidade real"
)

plt.plot(
    estimativas[:, 1],
    label="Velocidade estimada"
)

plt.xlabel("Tempo")
plt.ylabel("Velocidade")
plt.legend()
plt.grid()

plt.show()