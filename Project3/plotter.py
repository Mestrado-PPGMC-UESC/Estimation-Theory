import matplotlib.pyplot as plt


class Plotador:

    @staticmethod
    def plotar(I, S, R):

        tempo = range(len(I))

        plt.figure(figsize=(10, 6))

        plt.plot(tempo, I, label="Ignorantes")
        plt.plot(tempo, S, label="Espalhadores")
        plt.plot(tempo, R, label="Refutadores")

        plt.title("Propagação de Informação")

        plt.xlabel("Tempo")

        plt.ylabel("População")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plotar_convergencia(historico_beta, historico_alpha, historico_k, historico_erro, titulo):

        iteracoes = range(len(historico_erro))

        # Erro
        plt.figure(figsize=(10, 6))

        plt.plot(iteracoes, historico_erro, marker="o")

        plt.title(f"Convergência do Erro - {titulo}")

        plt.xlabel("Iteração")

        plt.ylabel("Norma do Resíduo")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

        # Beta
        plt.figure(figsize=(10, 6))

        plt.plot(iteracoes, historico_beta, marker="o")

        plt.title(f"Convergência de beta - {titulo}")

        plt.xlabel("Iteração")

        plt.ylabel("beta")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

        # Alpha
        plt.figure(figsize=(10, 6))

        plt.plot(iteracoes, historico_alpha, marker="o")

        plt.title(f"Convergência de alpha - {titulo}")

        plt.xlabel("Iteração")

        plt.ylabel("alpha")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

        # K
        plt.figure(figsize=(10, 6))

        plt.plot(iteracoes, historico_k, marker="o")

        plt.title(f"Convergência de k - {titulo}")

        plt.xlabel("Iteração")

        plt.ylabel("k")

        plt.grid(True)

        plt.tight_layout()

        plt.show()