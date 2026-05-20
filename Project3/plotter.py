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
    def plotar_convergencia(
        historico_beta,
        historico_alpha,
        historico_k,
        historico_delta,
        titulo
    ):

        iteracoes = range(len(historico_delta))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # -----------------------------
        # Delta
        # -----------------------------
        axs[0, 0].plot(iteracoes, historico_delta, marker="o")

        axs[0, 0].set_title(r"Norma de $\Delta \theta$")

        axs[0, 0].set_xlabel("Iteração")

        axs[0, 0].set_ylabel(r"$||\Delta \theta||$")

        axs[0, 0].set_yscale("log")

        axs[0, 0].grid(True)

        # -----------------------------
        # Beta
        # -----------------------------
        axs[0, 1].plot(iteracoes, historico_beta, marker="o")

        axs[0, 1].set_title(r"Convergência de $\beta$")

        axs[0, 1].set_xlabel("Iteração")

        axs[0, 1].set_ylabel(r"$\beta$")

        axs[0, 1].grid(True)

        # -----------------------------
        # Alpha
        # -----------------------------
        axs[1, 0].plot(iteracoes, historico_alpha, marker="o")

        axs[1, 0].set_title(r"Convergência de $\alpha$")

        axs[1, 0].set_xlabel("Iteração")

        axs[1, 0].set_ylabel(r"$\alpha$")

        axs[1, 0].grid(True)

        # -----------------------------
        # K
        # -----------------------------
        axs[1, 1].plot(iteracoes, historico_k, marker="o")

        axs[1, 1].set_title(r"Convergência de $k$")

        axs[1, 1].set_xlabel("Iteração")

        axs[1, 1].set_ylabel(r"$k$")

        axs[1, 1].grid(True)

        fig.suptitle(
            f"Convergência dos Parâmetros - {titulo}",
            fontsize=16
        )

        plt.tight_layout()

        plt.show()


    @staticmethod
    def plotar_mu(historico_mu, titulo):

        iteracoes = range(len(historico_mu))

        plt.figure(figsize=(10, 6))

        plt.plot(iteracoes, historico_mu, marker="o")

        plt.title(f"Evolução do Fator de Amortecimento - {titulo}")

        plt.xlabel("Iteração")

        plt.ylabel(r"$\mu$")

        plt.yscale("log")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plotar_delta_theta_comparacao(
        historico_delta_gn,
        historico_delta_lm
    ):

        iteracoes_gn = range(len(historico_delta_gn))

        iteracoes_lm = range(len(historico_delta_lm))

        plt.figure(figsize=(10, 6))

        plt.plot(
            iteracoes_gn,
            historico_delta_gn,
            marker="o",
            label="Gauss-Newton"
        )

        plt.plot(
            iteracoes_lm,
            historico_delta_lm,
            marker="s",
            label="Levenberg-Marquardt"
        )

        plt.yscale("log")

        plt.xlabel("Iteração")

        plt.ylabel(r"$||\Delta \theta||$")

        plt.title(r"Comparação da Convergência de $||\Delta \theta||$")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()


    @staticmethod
    def plotar_analise_mu(
        mus,
        historico_iteracoes,
        historico_delta_final
    ):

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # ---------------------------------
        # Iterações x mu
        # ---------------------------------

        axs[0].plot(
            mus,
            historico_iteracoes,
            marker="o"
        )

        axs[0].set_xscale("log")

        axs[0].set_title(r"Iterações vs $\mu_0$")

        axs[0].set_xlabel(r"$\mu_0$")

        axs[0].set_ylabel("Número de Iterações")

        axs[0].grid(True)

        # ---------------------------------
        # Delta final x mu
        # ---------------------------------

        axs[1].plot(
            mus,
            historico_delta_final,
            marker="o"
        )

        axs[1].set_xscale("log")

        axs[1].set_yscale("log")

        axs[1].set_title(r"$||\Delta \theta||$ final vs $\mu_0$")

        axs[1].set_xlabel(r"$\mu_0$")

        axs[1].set_ylabel(r"$||\Delta \theta||$ final")

        axs[1].grid(True)

        fig.suptitle(
            r"Influência do Valor Inicial de $\mu$",
            fontsize=16
        )

        plt.tight_layout()

        plt.show()