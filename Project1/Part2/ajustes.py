import numpy as np

def ajustar_linear(x, y):
    H = np.column_stack((x, np.ones(len(x))))

    theta = np.linalg.inv(H.T @ H) @ H.T @ y

    a = theta[0]
    b = theta[1]

    return a, b


def ajustar_quadratico(x, y):
    H = np.column_stack((x**2, x, np.ones(len(x))))

    theta = np.linalg.inv(H.T @ H) @ H.T @ y

    a = theta[0]
    b = theta[1]
    c = theta[2]

    return a, b, c


def ajustar_cubico(x, y):
    H = np.column_stack((x**3,sx**2, x, np.ones(len(x))))

    theta = np.linalg.inv(H.T @ H) @ H.T @ y

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]

    return a, b, c, d


def ajustar_linear_restrito(x, y):
    H = np.column_stack((x, np.ones(len(x))))

    A = H.T @ H
    b = H.T @ y

    C = np.array([[163, 1]])
    d = np.array([146])

    sistema = np.block([
        [A, C.T],
        [C, np.zeros((1, 1))]
    ])

    vetor = np.concatenate((b, d))

    solucao = np.linalg.solve(sistema, vetor)

    a = solucao[0]
    b_coef = solucao[1]
    

    return a, b_coef


def ajustar_quadratico_restrito(x, y):
    H = np.column_stack((x**2, x, np.ones(len(x))))

    A = H.T @ H
    b = H.T @ y

    C = np.array([[163**2, 163,1]])
    d = np.array([146])

    sistema = np.block([
        [A, C.T],
        [C, np.zeros((1, 1))]
    ])

    vetor = np.concatenate((b, d))

    solucao = np.linalg.solve(sistema, vetor)

    a = solucao[0]
    b_coef = solucao[1]
    c = solucao[2]

    return a, b_coef, c