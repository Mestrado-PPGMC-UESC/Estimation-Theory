import numpy as np
import matplotlib.pyplot as plt


def calcular_nrmse(referencia, aproximacao):

    rmse = np.sqrt(
        np.mean(
            (referencia - aproximacao)**2
        )
    )

    media = np.mean(
        np.abs(referencia)
    )

    return rmse / media