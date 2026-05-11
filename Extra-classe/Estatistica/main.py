import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Variáveis aleatórias medidas
# Exemplo: corrente e tensão
# -----------------------------
X = np.array([2.1, 2.0, 2.2, 1.9, 2.1])   # Corrente (A)
Y = np.array([12.2, 11.8, 12.5, 11.5, 12.0])  # Tensão (V)

# -----------------------------
# Média (valor esperado)
# -----------------------------
E_X = np.mean(X)
E_Y = np.mean(Y)

# -----------------------------
# Variância
# -----------------------------
Var_X = np.mean((X - E_X)**2)
Var_Y = np.mean((Y - E_Y)**2)

# -----------------------------
# Desvio padrão
# -----------------------------
Std_X = np.sqrt(Var_X)
Std_Y = np.sqrt(Var_Y)

# -----------------------------
# Covariância
# -----------------------------
Cov_XY = np.mean((X - E_X) * (Y - E_Y))
Cov_YX = np.mean((Y - E_Y) * (X - E_X))

# -----------------------------
# Matriz de covariância
# -----------------------------
P = np.array([
    [Var_X, Cov_XY],
    [Cov_YX, Var_Y]
])

# -----------------------------
# Resultados
# -----------------------------
print("=== MÉDIAS ===")
print(f"E[X] = {E_X:.4f}")
print(f"E[Y] = {E_Y:.4f}")

print("\n=== VARIÂNCIAS ===")
print(f"Var(X) = {Var_X:.4f}")
print(f"Var(Y) = {Var_Y:.4f}")

print("\n=== DESVIOS PADRÃO ===")
print(f"sigma(X) = {Std_X:.4f}")
print(f"sigma(Y) = {Std_Y:.4f}")

print("\n=== COVARIÂNCIA ===")
print(f"Cov(X,Y) = {Cov_XY:.4f}")
print(f"Cov(Y,X) = {Cov_YX:.4f}")

print("\n=== MATRIZ DE COVARIÂNCIA ===")
print(P)

# -----------------------------
# Distribuição Gaussiana
# -----------------------------
x_plot = np.linspace(E_X - 4*Std_X, E_X + 4*Std_X, 100)

gauss_x = (1 / (Std_X * np.sqrt(2*np.pi))) * np.exp(
    -0.5 * ((x_plot - E_X)/Std_X)**2
)

plt.figure()
plt.hist(X, bins=5, density=True, alpha=0.7, label="Dados")
plt.plot(x_plot, gauss_x, linewidth=2, label="Gaussiana")

plt.xlabel("Corrente (A)")
plt.ylabel("Densidade")
plt.title("Distribuição Gaussiana de X")
plt.legend()
plt.grid()
plt.show()