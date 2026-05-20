import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Dados
# x = jogos necessários
# y = gols atingidos
# -----------------------------
jogos_messi = np.array([0,213,313,417,524,631,746,862,1016,1142])
jogos_crist = np.array([0,301,457,553,653,752,854,973,1094,1235])

gols = np.linspace(0,900,10)

# -----------------------------
# MMQ polinomial cúbico
# gols = a3*x³ + a2*x² + a1*x + a0
# -----------------------------
coef_messi = np.polyfit(jogos_messi, gols, 3)
coef_crist = np.polyfit(jogos_crist, gols, 3)

p_messi = np.poly1d(coef_messi)
p_crist = np.poly1d(coef_crist)

# derivadas
dp_messi = np.polyder(p_messi)
dp_crist = np.polyder(p_crist)

# malha fina
x_messi = np.linspace(
    jogos_messi.min(),
    jogos_messi.max(),
    500
)

x_crist = np.linspace(
    jogos_crist.min(),
    jogos_crist.max(),
    500
)

# taxas suavizadas
taxa_messi = dp_messi(x_messi)
taxa_crist = dp_crist(x_crist)

# evita taxa negativa causada pelo ajuste
taxa_messi = np.maximum(taxa_messi, 0)
taxa_crist = np.maximum(taxa_crist, 0)

# -----------------------------
# 1. Dados + ajuste cúbico
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.scatter(jogos_messi, gols, label="Dados Messi")
plt.plot(x_messi, p_messi(x_messi), label="MMQ cúbico")
plt.xlabel("Jogos")
plt.ylabel("Gols")
plt.title("Messi")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(jogos_crist, gols, label="Dados Cristiano")
plt.plot(x_crist, p_crist(x_crist), label="MMQ cúbico")
plt.xlabel("Jogos")
plt.ylabel("Gols")
plt.title("Cristiano")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 2. Taxa contínua
# -----------------------------
plt.figure(figsize=(8,4))

plt.plot(
    x_messi,
    taxa_messi,
    label="Messi"
)

plt.plot(
    x_crist,
    taxa_crist,
    label="Cristiano"
)

plt.xlabel("Jogos")
plt.ylabel("Taxa (gols/jogo)")
plt.title("Evolução da taxa de gols")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Taxa inicial real
# -----------------------------
taxa_inicial_messi = 100 / (jogos_messi[1] - jogos_messi[0])
taxa_inicial_crist = 100 / (jogos_crist[1] - jogos_crist[0])

print(
    f"Messi nos primeiros 100 gols: "
    f"{taxa_inicial_messi:.4f} gols/jogo"
)

print(
    f"Cristiano nos primeiros 100 gols: "
    f"{taxa_inicial_crist:.4f} gols/jogo"
)

print(
    f"Messi em 100 jogos (curva): "
    f"{dp_messi(100):.4f} gols/jogo"
)

print(
    f"Cristiano em 100 jogos (curva): "
    f"{dp_crist(100):.4f} gols/jogo"
)

# -----------------------------
# Previsão para 1000 gols
# usando busca na curva
# -----------------------------
meta = 1000

x_messi_prev = np.linspace(jogos_messi.min(), 2000, 5000)
x_crist_prev = np.linspace(jogos_crist.min(), 2000, 5000)

y_messi_prev = p_messi(x_messi_prev)
y_crist_prev = p_crist(x_crist_prev)

idx_messi = np.argmin(np.abs(y_messi_prev - meta))
idx_crist = np.argmin(np.abs(y_crist_prev - meta))

jogos_1000_messi = x_messi_prev[idx_messi]
jogos_1000_crist = x_crist_prev[idx_crist]

print(
    f"Messi chega aos 1000 gols em aproximadamente "
    f"{jogos_1000_messi:.2f} jogos"
)

print(
    f"Cristiano chega aos 1000 gols em aproximadamente "
    f"{jogos_1000_crist:.2f} jogos"
)

# -----------------------------
# Taxa real interpolada
# -----------------------------
taxa_local_messi = np.diff(gols) / np.diff(jogos_messi)
taxa_local_crist = np.diff(gols) / np.diff(jogos_crist)

# centro de cada intervalo
jogos_meio_messi = (
    jogos_messi[:-1] + jogos_messi[1:]
) / 2

jogos_meio_crist = (
    jogos_crist[:-1] + jogos_crist[1:]
) / 2

plt.figure(figsize=(8,4))

plt.plot(
    jogos_meio_messi,
    taxa_local_messi,
    'o-',
    label="Messi"
)

plt.plot(
    jogos_meio_crist,
    taxa_local_crist,
    'o-',
    label="Cristiano"
)

plt.xlabel("Jogos")
plt.ylabel("Taxa real (gols/jogo)")
plt.title("Taxa real interpolada entre pontos")
plt.legend()
plt.grid(True)
plt.show()