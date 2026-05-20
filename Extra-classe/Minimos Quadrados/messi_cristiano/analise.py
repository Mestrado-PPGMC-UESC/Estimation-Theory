import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_messi = pd.read_csv("messi_jogos_todos.csv")
df_crist = pd.read_csv("cristiano_jogos_todos.csv")
df_romario = pd.read_csv("romario_jogos_todos.csv")
df_pele = pd.read_csv("pele_jogos_todos.csv")
df_ronaldo = pd.read_csv("ronaldo_jogos_todos.csv")
df_mbappe = pd.read_csv("mbappe_jogos_todos.csv")
df_eusebio = pd.read_csv("eusebio_jogos_todos.csv")


# -----------------------------
# Dados
# -----------------------------
jogos_messi = df_messi["jogo"].to_numpy()
gols_messi = df_messi["gols_acumulados"].to_numpy()

jogos_crist = df_crist["jogo"].to_numpy()
gols_crist = df_crist["gols_acumulados"].to_numpy()

jogos_romario = df_romario["jogo"].to_numpy()
gols_romario = df_romario["gols_acumulados"].to_numpy()

jogos_pele = df_pele["jogo"].to_numpy()
gols_pele = df_pele["gols_acumulados"].to_numpy()

jogos_ronaldo = df_ronaldo["jogo"].to_numpy()
gols_ronaldo = df_ronaldo["gols_acumulados"].to_numpy()

jogos_mbappe = df_mbappe["jogo"].to_numpy()
gols_mbappe = df_mbappe["gols_acumulados"].to_numpy()

jogos_eusebio = df_eusebio["jogo"].to_numpy()
gols_eusebio = df_eusebio["gols_acumulados"].to_numpy()


# -----------------------------
# 1. Gols acumulados
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(jogos_messi, gols_messi, label="Messi")
plt.plot(jogos_crist, gols_crist, label="Cristiano")
plt.plot(jogos_romario, gols_romario, label="Romário")
plt.plot(jogos_pele, gols_pele, label="Pelé")
plt.plot(jogos_ronaldo, gols_ronaldo, label="Ronaldo")
plt.plot(jogos_mbappe, gols_mbappe, label="Mbappé")
plt.plot(jogos_eusebio, gols_eusebio, label="Eusébio")

plt.xlabel("Jogos")
plt.ylabel("Gols acumulados")
plt.title("Evolução de gols acumulados")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 2. Taxa real suavizada
# -----------------------------
janela = 100

taxa_messi = df_messi["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_crist = df_crist["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_romario = df_romario["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_pele = df_pele["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_ronaldo = df_ronaldo["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_mbappe = df_mbappe["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()
taxa_eusebio = df_eusebio["gols_no_jogo"].rolling(window=janela, min_periods=1).mean()

plt.figure(figsize=(10,5))

plt.plot(jogos_messi, taxa_messi, label="Messi")
plt.plot(jogos_crist, taxa_crist, label="Cristiano")
plt.plot(jogos_romario, taxa_romario, label="Romário")
plt.plot(jogos_pele, taxa_pele, label="Pelé")
plt.plot(jogos_ronaldo, taxa_ronaldo, label="Ronaldo")
plt.plot(jogos_mbappe, taxa_mbappe, label="Mbappé")
plt.plot(jogos_eusebio, taxa_eusebio, label="Eusébio")

plt.xlabel("Jogos")
plt.ylabel("Gols por jogo")
plt.title(f"Taxa de gols suavizada - média móvel ({janela} jogos)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Marcos de carreira
# -----------------------------
metas = np.arange(
    100,
    min(
        gols_messi.max(),
        gols_crist.max(),
        gols_romario.max(),
        gols_pele.max(),
        gols_ronaldo.max(),
        gols_mbappe.max(),
        gols_eusebio.max()
    ) + 1,
    100
)

print("\nMarcos de carreira")
print("--------------------------------------------------------------------------------------------------------")

for meta in metas:

    idx_messi = np.where(gols_messi >= meta)[0][0]
    idx_crist = np.where(gols_crist >= meta)[0][0]
    idx_romario = np.where(gols_romario >= meta)[0][0]
    idx_pele = np.where(gols_pele >= meta)[0][0]
    idx_ronaldo = np.where(gols_ronaldo >= meta)[0][0]
    idx_mbappe = np.where(gols_mbappe >= meta)[0][0]
    idx_eusebio = np.where(gols_eusebio >= meta)[0][0]

    print(
        f"{meta} gols | "
        f"Messi: {jogos_messi[idx_messi]} | "
        f"Cristiano: {jogos_crist[idx_crist]} | "
        f"Romário: {jogos_romario[idx_romario]} | "
        f"Pelé: {jogos_pele[idx_pele]} | "
        f"Ronaldo: {jogos_ronaldo[idx_ronaldo]} | "
        f"Mbappé: {jogos_mbappe[idx_mbappe]} | "
        f"Eusébio: {jogos_eusebio[idx_eusebio]}"
    )