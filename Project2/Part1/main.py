from model import ModeloPreco
from config import (
    taxa_crescimento,
    volatilidade,
    passo_tempo,
    ruido_observacao,
    tendencia_sistematica,
    numero_passos,
    estado_inicial,
    semente_aleatoria
)
from simulacao import gerar_dados
from analises import analise_inicial_ruido,analise_ruido_processo,analise_multiplas_sementes,analise_inicial_observacoes,analise_multiplas_sementes_observacao,analise_kalman_filtrado_multiplas_sementes,analise_kalman_preditivo_multiplas_sementes,analise_kalman_preditivo_corretivo_multiplas_sementes

from kalman import estimativa_filtrada
from plots.plot_kalman import plot_kalman_comparacao

def main():



    modelo = ModeloPreco(
        taxa_crescimento,
        volatilidade,
        passo_tempo,
        ruido_observacao,
        tendencia_sistematica
    )

    trajetoria_sem_ruido,trajetoria_com_ruido,observacoes = gerar_dados(numero_passos, estado_inicial, modelo)
    analise_inicial_ruido(trajetoria_sem_ruido,trajetoria_com_ruido)
    analise_ruido_processo()
    analise_multiplas_sementes(numero_passos, estado_inicial, modelo,1000)
    analise_inicial_observacoes(trajetoria_com_ruido,observacoes)
    analise_multiplas_sementes_observacao(numero_passos, estado_inicial, modelo,1000)


    analise_kalman_filtrado_multiplas_sementes(
        numero_passos,
        estado_inicial,
        modelo,
        numero_iteracoes=1000,
        seed_plot=42
    )

    analise_kalman_preditivo_multiplas_sementes(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
)
    

    analise_kalman_preditivo_corretivo_multiplas_sementes(
    numero_passos,
    estado_inicial,
    modelo,
    numero_iteracoes=1000,
    seed_plot=42
)

if __name__ == "__main__":
    main()