import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from estimador_filtrado import estimar_filtrado
from avaliar_erro import calcular_erro_filtrado


F_global = None
H_global = None
medicoes_global = None


def inicializar_worker(F, H, medicoes):

    global F_global
    global H_global
    global medicoes_global

    F_global = F
    H_global = H
    medicoes_global = medicoes


def avaliar_par(parametros):

    i, j, q_valor, r_valor = parametros

    Q = q_valor * np.eye(3)

    R = np.array([
        [r_valor]
    ])

    numero_passos = len(medicoes_global)

    estimativas_filtradas, _ = estimar_filtrado(
        F=F_global,
        H=H_global,
        Q=Q,
        R=R,
        medicoes=medicoes_global,
        numero_passos=numero_passos
    )

    erro_percentual = calcular_erro_filtrado(
        estimativas_filtradas=estimativas_filtradas
    )

    return i, j, erro_percentual


def montar_plano_q_r(F, H, medicoes):

    valores_q = np.arange(0.1, 50, 0.1)
    valores_r = np.arange(0.1, 50, 0.1)

    parametros = []

    for i, q_valor in enumerate(valores_q):
        for j, r_valor in enumerate(valores_r):
            parametros.append((i, j, q_valor, r_valor))

    matriz_erro = np.zeros((len(valores_q), len(valores_r)))

    print(f"\nUsando {cpu_count()} núcleos...")
    print(f"Total de combinações: {len(parametros)}")

    with Pool(processes=cpu_count(), initializer=inicializar_worker, initargs=(F, H, medicoes)) as pool:
        resultados = pool.map(avaliar_par, parametros)

    for i, j, erro in resultados:
        matriz_erro[i, j] = erro

    indice_menor_erro = np.unravel_index(np.argmin(matriz_erro), matriz_erro.shape)

    melhor_q = valores_q[indice_menor_erro[0]]
    melhor_r = valores_r[indice_menor_erro[1]]
    menor_erro = matriz_erro[indice_menor_erro]

    print("\nResultado da busca")
    print("--------------------------------")
    print(f"Melhor Q: {melhor_q:.4f}")
    print(f"Melhor R: {melhor_r:.4f}")
    print(f"Menor erro: {menor_erro:.6f} %")

    R_grid, Q_grid = np.meshgrid(valores_r, valores_q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(R_grid, Q_grid, matriz_erro)

    ax.scatter(melhor_r, melhor_q, menor_erro, s=100, marker="o", color="red", label="Menor erro")

    ax.set_xlabel("R")
    ax.set_ylabel("Q")
    ax.set_zlabel("Erro (%)")
    ax.set_title("Superfície de erro em função de Q e R")
    ax.legend()

    plt.show()

    return valores_q, valores_r, matriz_erro, melhor_q, melhor_r, menor_erro