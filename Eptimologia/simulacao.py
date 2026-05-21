import matplotlib.pyplot as plt
from metrics import calcular_nrmse

def executar(modelo):
    return modelo.simular()


def executar_varios(modelos):
    resultados = {}

    for nome, modelo in modelos.items():
        resultados[nome] = executar(modelo)

    return resultados


def plotar_infectados(resultados):
    plt.figure(figsize=(10, 5))

    for nome, resultado in resultados.items():
        t, S, I = resultado
        plt.plot(t, I, linewidth=2, label=nome)

    plt.title("Modelo SIS: não linear vs linearizado")
    plt.xlabel("Tempo")
    plt.ylabel("Infectados")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotar_suscetiveis(resultados):
    plt.figure(figsize=(10, 5))

    for nome, resultado in resultados.items():
        t, S, I = resultado
        plt.plot(t, S, linewidth=2, label=nome)

    plt.title("Suscetíveis no modelo SIS")
    plt.xlabel("Tempo")
    plt.ylabel("Suscetíveis")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotar_erro(resultados):

    t_ref, S_ref, I_ref = resultados["SIS não linear"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8)
    )

    for nome, (t, S, I) in resultados.items():

        if nome == "SIS não linear":
            continue

        erro_S = S_ref - S
        erro_I = I_ref - I

        ax1.plot(
            t,
            erro_I,
            linewidth=2,
            label=nome
        )

        ax2.plot(
            t,
            erro_S,
            linewidth=2,
            label=nome
        )

        nrmse_I = calcular_nrmse(
            I_ref,
            I
        )

        nrmse_S = calcular_nrmse(
            S_ref,
            S
        )

        print(f"\n{nome}")
        print(f"NRMSE Infectados: {nrmse_I:.6f}")
        print(f"NRMSE Suscetíveis: {nrmse_S:.6f}")


    ax1.set_title(
        "Erro entre modelos - Infectados"
    )

    ax1.set_ylabel(
        "Erro"
    )

    ax1.grid(True)
    ax1.legend()


    ax2.set_title(
        "Erro entre modelos - Suscetíveis"
    )

    ax2.set_xlabel(
        "Tempo"
    )

    ax2.set_ylabel(
        "Erro"
    )

    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()