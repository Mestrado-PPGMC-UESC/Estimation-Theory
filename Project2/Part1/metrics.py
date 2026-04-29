import numpy as np


def calcular_rmse(valores, referencia):
    return np.sqrt(np.mean((referencia - valores)**2))


def calcular_nrmse(valores, referencia):
    rmse = calcular_rmse(valores, referencia)
    return rmse / np.mean(referencia)