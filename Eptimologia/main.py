from ModeloSISNaoLinear import ModeloSISNaoLinear
from ModeloSISPontoEstavel import ModeloSISPontoEstavel
from ModeloSISPontoInstavel import ModeloSISPontoInstavel

from simulacao import (
    executar_varios,
    plotar_infectados,
    plotar_suscetiveis,
    plotar_erro
)



def main():

    alpha = 0.9      # taxa de infecção
    beta = 0.1429    # taxa de recuperação
    gamma = 0.0      # taxa de mortalidade/natalidade

    N = 1.0

    S0 = 0.9
    I0 = 0.1

    # passo inicial para gráfico estático
    h = 0.00001

    T = 10


    modelos = {

        "SIS não linear": ModeloSISNaoLinear(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            N=N,
            S0=S0,
            I0=I0,
            passo_tempo=h,
            tempo_total=T
        ),

        "Linearizado - ponto estável": ModeloSISPontoEstavel(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            N=N,
            S0=S0,
            I0=I0,
            passo_tempo=h,
            tempo_total=T
        )
    }


    # gráficos estáticos
    resultados = executar_varios(modelos)

    plotar_infectados(resultados)
    plotar_suscetiveis(resultados)


    plotar_erro(resultados)



if __name__ == "__main__":
    main()