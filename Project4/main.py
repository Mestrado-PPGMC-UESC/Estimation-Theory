import numpy as np

from model import ModeloPropagacaoInformacao
from simulator import Simulador
from kalman_extendido import FiltroKalmanExtendido
from plotter import Plotador

from config import beta_real, alpha_real, k_real, numero_passos, I0, S0, R0, I0_estimado, S0_estimado, R0_estimado, Q, R, P0, sigma_observacao


def main():

    # -----------------------------
    # 1. Modelo real
    # -----------------------------

    modelo_real = ModeloPropagacaoInformacao(beta=beta_real, alpha=alpha_real, k=k_real)

    # -----------------------------
    # 2. Simulação real
    # -----------------------------

    simulador_real = Simulador(modelo=modelo_real, numero_passos=numero_passos)

    I_real, S_real, R_real = simulador_real.executar(I0=I0, S0=S0, R0=R0)

    # -----------------------------
    # 3. Organiza os estados reais
    # -----------------------------

    estados_reais = np.column_stack((I_real, S_real, R_real))

    # -----------------------------
    # 4. Adiciona ruído nas medições
    # -----------------------------

    ruido_observacao = np.random.normal(loc=0.0, scale=sigma_observacao, size=estados_reais.shape)

    observacoes = estados_reais + ruido_observacao

    # -----------------------------
    # 5. Cria o filtro
    # -----------------------------

    filtro_ekf = FiltroKalmanExtendido(modelo=modelo_real, Q=Q, R=R, P0=P0)

    # -----------------------------
    # 6. Estado inicial estimado
    # -----------------------------

    estado_inicial_estimado = np.array([I0_estimado, S0_estimado, R0_estimado])

    # -----------------------------
    # 7. Executa o filtro
    # -----------------------------

    estados_estimados = filtro_ekf.filtrar(observacoes=observacoes, estado_inicial=estado_inicial_estimado)

    # -----------------------------
    # 8. Separa os estados estimados
    # -----------------------------

    I_estimado = estados_estimados[:, 0]
    S_estimado = estados_estimados[:, 1]
    R_estimado = estados_estimados[:, 2]

    # -----------------------------
    # 9. Plota resultados
    # -----------------------------

    Plotador.plotar_kalman(I_real, S_real, R_real, I_estimado, S_estimado, R_estimado)


if __name__ == "__main__":
    main()