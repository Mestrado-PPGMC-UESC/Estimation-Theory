import numpy as np

def calcular_nrmse(y_real, y_estimado):

    erro = y_real - y_estimado

    rmse = np.sqrt(
        np.mean(
            erro**2
        )
    )

    nrmse = rmse / np.mean(y_real)

    return nrmse