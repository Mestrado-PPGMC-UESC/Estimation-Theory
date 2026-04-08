import numpy as np

def calcular_erros(y_real, y_estimado):
    erro = y_real - y_estimado

    erros_relativos = np.abs(erro) / (np.abs(y_real) + 1e-16)
    rmse = np.sqrt(np.mean(erro**2))

    print('Erros relativos por ponto:')
    for i, erro_relativo in enumerate(erros_relativos):
        print(f'Ponto {i + 1}: {erro_relativo:.4f}')

    print(f'RMSE: {rmse:.4f}')


def calcular_previsao_linear(x, a, b):
    return a * x + b


def calcular_previsao_quadratica(x, a, b, c):
    return a * x**2 + b * x + c


def comparacao(renda,consumo,a1,b1,a2,b2,c2):

    # Previsões
    ajuste_linear_consumo = calcular_previsao_linear(renda, a1, b1)
    ajuste_quadratico_consumo = calcular_previsao_quadratica(renda , a2, b2, c2)

    # Equações
    print('Ajuste Linear:')
    print(f'y = {a1:.4f}x + {b1:.4f}')

    print()

    print('Ajuste Quadrático:')
    print(f'y = {a2:.4f}x² + {b2:.4f}x + {c2:.4f}')

    print()
    print('Ajuste Linear')
    calcular_erros(consumo, ajuste_linear_consumo)

    print()
    print('Ajuste Quadrático')
    calcular_erros(consumo, ajuste_quadratico_consumo)

