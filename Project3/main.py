import numpy as np

from model import ModeloPropagacaoInformacao
from simulator import Simulador
from plotter import Plotador
from gauss_newton import EstimadorGaussNewton
from levenberg_marquardt import EstimadorLevenbergMarquardt
from levenberg_marquardt_adaptativo import EstimadorLevenbergMarquardtAdaptativo
from config import (beta_real,alpha_real,k_real,numero_passos,I0,S0,R0,beta_chute,alpha_chute,k_chute,mu_fixo,mus)


def main():


    # -----------------------------
    # 1. Simulação do modelo
    # -----------------------------

    modelo_real = ModeloPropagacaoInformacao(beta=beta_real,alpha=alpha_real,k=k_real)

    simulador_real = Simulador(modelo=modelo_real,numero_passos=numero_passos)

    I_real, S_real, R_real = simulador_real.executar(I0=I0,S0=S0,R0=R0)

    dados_observados = np.column_stack((I_real,S_real,R_real))

    Plotador.plotar(I_real,S_real,R_real)

    # -----------------------------
    # 2. Estimação por Gauss-Newton
    # -----------------------------

    estimador = EstimadorGaussNewton(numero_passos=numero_passos,I0=I0,S0=S0,R0=R0,dados_observados=dados_observados)

    parametros_estimados,historico_beta,historico_alpha,historico_k,historico_erro,iteracoes= estimador.estimar(parametros_iniciais=[beta_chute, alpha_chute, k_chute])

    beta_estimado, alpha_estimado, k_estimado = parametros_estimados

    print(beta_estimado,alpha_estimado,k_estimado)

    # -----------------------------
    # 3. Resultados
    # -----------------------------

    print(
        f"\nResultados da estimação - Gauss-Newton\n"
        f"--------------------------------------\n"
        f"{'Beta real:':<20}{beta_real:.6f}\n"
        f"{'Beta estimado:':<20}{beta_estimado:.6f}\n\n"
        f"{'Alpha real:':<20}{alpha_real:.6f}\n"
        f"{'Alpha estimado:':<20}{alpha_estimado:.6f}\n\n"
        f"{'K real:':<20}{k_real:.6f}\n"
        f"{'K estimado:':<20}{k_estimado:.6f}\n\n"
        f"{'Iterações:':<20}{iteracoes}"
    )

    # -----------------------------
    # 4. Modelo estimado
    # -----------------------------

    modelo_estimado = ModeloPropagacaoInformacao(beta=beta_estimado,alpha=alpha_estimado,k=k_estimado,)

    simulador_estimado = Simulador(modelo=modelo_estimado,numero_passos=numero_passos)

    I_est, S_est, R_est = simulador_estimado.executar(I0=I0,S0=S0,R0=R0)

    Plotador.plotar(I_est,S_est,R_est)
    
    # -----------------------------
    # 5. Convergência - Gauss-Newton
    # -----------------------------

    Plotador.plotar_convergencia(historico_beta, historico_alpha, historico_k, historico_erro, titulo="Gauss-Newton")

    # -----------------------------
    # 6. Levenberg-Marquardt
    # -----------------------------

    parametros_iniciais_lm = [beta_chute, alpha_chute, k_chute]

    estimador_lm = EstimadorLevenbergMarquardt(numero_passos=numero_passos, I0=I0, S0=S0, R0=R0, dados_observados=dados_observados, lamb=mu_fixo)


    parametros_lm,historico_beta_lm,historico_alpha_lm,historico_k_lm,historico_erro_lm,iteracoes_lm,= estimador_lm.estimar(parametros_iniciais=parametros_iniciais_lm)

    beta_lm, alpha_lm, k_lm = parametros_lm

    produto_real = beta_real * k_real
    produto_lm = beta_lm * k_lm

    print(
        f"\nResultados da estimação - Levenberg-Marquardt\n"
        f"---------------------------------------------\n"
        f"{'Beta real:':<28}{beta_real:.6f}\n"
        f"{'Beta estimado:':<28}{beta_lm:.6f}\n\n"
        f"{'Alpha real:':<28}{alpha_real:.6f}\n"
        f"{'Alpha estimado:':<28}{alpha_lm:.6f}\n\n"
        f"{'K real:':<28}{k_real:.6f}\n"
        f"{'K estimado:':<28}{k_lm:.6f}\n\n"
        f"{'Beta*k real:':<28}{produto_real:.6f}\n"
        f"{'Beta*k estimado:':<28}{produto_lm:.6f}\n\n"
        f"{'Lambda:':<28}{mu_fixo:.6e}\n"
        f"{'Iterações:':<28}{iteracoes_lm}"
    )

    Plotador.plotar_convergencia(historico_beta_lm, historico_alpha_lm, historico_k_lm, historico_erro_lm, titulo="Levenberg-Marquardt")

    # -----------------------------
    # 7. Teste com diferentes valores fixos de mu
    # -----------------------------

    print("\nTeste com diferentes valores de mu - Levenberg-Marquardt")
    print("------------------------------------------------------------------------------------------------")
    print("mu           iter   beta          alpha         k             beta*k        erro")
    print("------------------------------------------------------------------------------------------------")

    for mu in mus:

        estimador_lm = EstimadorLevenbergMarquardt(numero_passos=numero_passos, I0=I0, S0=S0, R0=R0, dados_observados=dados_observados, lamb=mu)

        (parametros_lm, historico_beta_lm, historico_alpha_lm, historico_k_lm, historico_erro_lm, iteracoes_lm) = estimador_lm.estimar(parametros_iniciais=parametros_iniciais_lm)

        beta_lm, alpha_lm, k_lm = parametros_lm

        produto_lm = beta_lm * k_lm

        erro_final_lm = historico_erro_lm[-1]

        print(
            f"{mu:<12.0e}"
            f"{iteracoes_lm:<7}"
            f"{beta_lm:<14.6f}"
            f"{alpha_lm:<14.6f}"
            f"{k_lm:<14.6f}"
            f"{produto_lm:<14.6f}"
            f"{erro_final_lm:.6e}"
        )

# 8. Levenberg-Marquardt Adaptativo
# -----------------------------

    estimador_lm_adaptativo = EstimadorLevenbergMarquardtAdaptativo(numero_passos=numero_passos, I0=I0, S0=S0, R0=R0, dados_observados=dados_observados, mu=mu_fixo)

    (parametros_lm_adaptativo, historico_beta_lm_adaptativo, historico_alpha_lm_adaptativo, historico_k_lm_adaptativo, historico_erro_lm_adaptativo, historico_mu_lm_adaptativo, iteracoes_lm_adaptativo) = estimador_lm_adaptativo.estimar(parametros_iniciais=parametros_iniciais_lm)

    beta_lm_adaptativo, alpha_lm_adaptativo, k_lm_adaptativo = parametros_lm_adaptativo

    produto_lm_adaptativo = beta_lm_adaptativo * k_lm_adaptativo

    print(
        f"\nResultados da estimação - Levenberg-Marquardt Adaptativo\n"
        f"--------------------------------------------------------\n"
        f"{'Beta real:':<28}{beta_real:.6f}\n"
        f"{'Beta estimado:':<28}{beta_lm_adaptativo:.6f}\n\n"
        f"{'Alpha real:':<28}{alpha_real:.6f}\n"
        f"{'Alpha estimado:':<28}{alpha_lm_adaptativo:.6f}\n\n"
        f"{'K real:':<28}{k_real:.6f}\n"
        f"{'K estimado:':<28}{k_lm_adaptativo:.6f}\n\n"
        f"{'Beta*k real:':<28}{produto_real:.6f}\n"
        f"{'Beta*k estimado:':<28}{produto_lm_adaptativo:.6f}\n\n"
        f"{'Mu inicial:':<28}{mu_fixo:.6e}\n"
        f"{'Mu final:':<28}{historico_mu_lm_adaptativo[-1]:.6e}\n"
        f"{'Iterações:':<28}{iteracoes_lm_adaptativo}"
    )

    Plotador.plotar_convergencia(historico_beta_lm_adaptativo, historico_alpha_lm_adaptativo, historico_k_lm_adaptativo, historico_erro_lm_adaptativo, titulo="LM Adaptativo")


    # 9. Teste com diferentes valores iniciais de mu adaptativo
    # -----------------------------



    print("\nTeste com diferentes valores iniciais de mu - LM Adaptativo")
    print("----------------------------------------------------------------------------------------------------------------")
    print("mu inicial   iter   beta          alpha         k             beta*k        mu final      erro")
    print("----------------------------------------------------------------------------------------------------------------")

    for mu in mus:

        estimador_lm_adaptativo = EstimadorLevenbergMarquardtAdaptativo(numero_passos=numero_passos, I0=I0, S0=S0, R0=R0, dados_observados=dados_observados, mu=mu)

        (parametros_lm_adaptativo, historico_beta_lm_adaptativo, historico_alpha_lm_adaptativo, historico_k_lm_adaptativo, historico_erro_lm_adaptativo, historico_mu_lm_adaptativo, iteracoes_lm_adaptativo) = estimador_lm_adaptativo.estimar(parametros_iniciais=parametros_iniciais_lm)

        beta_lm_adaptativo, alpha_lm_adaptativo, k_lm_adaptativo = parametros_lm_adaptativo

        produto_lm_adaptativo = beta_lm_adaptativo * k_lm_adaptativo

        mu_final = historico_mu_lm_adaptativo[-1]

        erro_final_lm_adaptativo = historico_erro_lm_adaptativo[-1]

        print(
            f"{mu:<13.0e}"
            f"{iteracoes_lm_adaptativo:<7}"
            f"{beta_lm_adaptativo:<14.6f}"
            f"{alpha_lm_adaptativo:<14.6f}"
            f"{k_lm_adaptativo:<14.6f}"
            f"{produto_lm_adaptativo:<14.6f}"
            f"{mu_final:<14.6e}"
            f"{erro_final_lm_adaptativo:.6e}"
        )

if __name__ == "__main__":
    main()