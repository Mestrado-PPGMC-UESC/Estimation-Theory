import numpy as np


# -----------------------------
# Parâmetros reais do modelo
# -----------------------------

beta_real = 0.04
alpha_real = 0.03
k_real = 0.8


# -----------------------------
# Configuração da simulação
# -----------------------------

numero_passos = 500

I0 = 0.9
S0 = 0.1
R0 = 0.0


# -----------------------------
# Chute inicial do EKF
# -----------------------------

I0_estimado = 0.001
S0_estimado = 0.33
R0_estimado = 0.33


# -----------------------------
# Matrizes do EKF
# -----------------------------

# Covariância do ruído de processo
Q = np.eye(3) * 1e-2

# Covariância do ruído de observação
R = np.eye(3) * 1e-2

# Covariância inicial da estimação
P0 = np.eye(3) * 1e-2


# -----------------------------
# Ruído artificial das medições
# -----------------------------

sigma_observacao = 0.01