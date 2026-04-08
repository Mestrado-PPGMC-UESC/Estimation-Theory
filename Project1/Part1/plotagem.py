import matplotlib.pyplot as plt
import numpy as np

from auxiliar import calcular_erro_absoluto

def criar_grafico(S, I, R, titulo, limite=None):
    plt.figure(figsize=(10, 6))

    if limite is not None:
        S = S[:limite]
        I = I[:limite]
        R = R[:limite]

    plt.plot(S, label='Suscetíveis')
    plt.plot(I, label='Infectados')
    plt.plot(R, label='Recuperados')

    plt.xlabel('Tempo')
    plt.ylabel('População')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)


def plotar_simulacao(S, I, R): 

    criar_grafico(S, I, R,titulo='Modelo SIR Discreto')

    criar_grafico(S, I, R,titulo='Modelo SIR Discreto - Primeiras 50 Iterações',limite=51)

    plt.show()


def plotar_comparacao(S, I, R, S_estimado, I_estimado, R_estimado):

    variaveis = [('Suscetíveis', S, S_estimado),('Infectados', I, I_estimado),('Recuperados', R, R_estimado)]

    for nome, original, estimado in variaveis:
        erro = calcular_erro_absoluto(original, estimado)

        plt.figure(figsize=(12, 5))

        # Curvas original e estimada
        plt.subplot(1, 2, 1)
        plt.plot(original, label=f'{nome} Original')
        plt.plot(estimado, '--', label=f'{nome} Estimado')
        plt.xlabel('Tempo')
        plt.ylabel('População')
        plt.title(f'Comparação - {nome}')
        plt.legend()
        plt.grid(True)

        # Erro absoluto
        plt.subplot(1, 2, 2)
        plt.plot(erro, label='Erro Relativo')
        plt.xlabel('Tempo')
        plt.ylabel('Erro')
        plt.title(f'Erro - {nome}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

    plt.show()

    
def plotar_comparacao_2(S, I, R,S_estimado, I_estimado, R_estimado,S_estimado_reduzido, I_estimado_reduzido, R_estimado_reduzido):
    variaveis = [
        ('Suscetíveis',S,S_estimado,S_estimado_reduzido),
        ('Infectados',I,I_estimado,I_estimado_reduzido),
        ('Recuperados',R,R_estimado,R_estimado_reduzido)
    ]

    for nome, original, estimado, estimado_reduzido in variaveis:
        erro_estimado = calcular_erro_absoluto(original, estimado)
        erro_reduzido = calcular_erro_absoluto(original, estimado_reduzido)

        plt.figure(figsize=(14, 5))

        # Comparação das curvas
        plt.subplot(1, 2, 1)
        plt.plot(original, label=f'{nome} Original', linewidth=2)
        plt.plot(estimado, '--', label='Estimado Completo')
        plt.plot(estimado_reduzido, ':', label='Estimado Reduzido')

        plt.xlabel('Tempo')
        plt.ylabel('População')
        plt.title(f'Comparação - {nome}')
        plt.legend()
        plt.grid(True)

        # Comparação dos erros relativos
        plt.subplot(1, 2, 2)
        plt.plot(erro_estimado, label='Erro Abosluto Completo')
        plt.plot(erro_reduzido, label='Erro Abosluto Reduzido')

        plt.xlabel('Tempo')
        plt.ylabel('Erro Abosluto')
        plt.title(f'Erro Abosluto - {nome}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

    plt.show()