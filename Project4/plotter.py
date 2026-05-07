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
    def plotar_kalman(I_real, S_real, R_real, I_estimado, S_estimado, R_estimado):

        tempo_real = range(len(I_real))
        tempo_estimado = range(1, len(I_estimado) + 1)

        plt.figure(figsize=(10, 6))

        plt.plot(tempo_real, I_real, label="I Real")
        plt.plot(tempo_estimado, I_estimado, linestyle="--", label="I Estimado")

        plt.title("Filtro de Kalman Estendido - Ignorantes")

        plt.xlabel("Tempo")
        plt.ylabel("População")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        plt.plot(tempo_real, S_real, label="S Real")
        plt.plot(tempo_estimado, S_estimado, linestyle="--", label="S Estimado")

        plt.title("Filtro de Kalman Estendido - Espalhadores")

        plt.xlabel("Tempo")
        plt.ylabel("População")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        plt.plot(tempo_real, R_real, label="R Real")
        plt.plot(tempo_estimado, R_estimado, linestyle="--", label="R Estimado")

        plt.title("Filtro de Kalman Estendido - Refutadores")

        plt.xlabel("Tempo")
        plt.ylabel("População")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()