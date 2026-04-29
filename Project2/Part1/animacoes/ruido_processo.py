import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from model import ModeloPreco
from simulacao import gerar_estado_sem_ruido, gerar_estado_com_ruido
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


def gerar_dados(delta_atual, q_atual=0.0):

    np.random.seed(semente_aleatoria)

    modelo = ModeloPreco(
        taxa_crescimento,
        delta_atual,
        passo_tempo,
        q_atual,
        tendencia_sistematica
    )

    trajetoria_sem_ruido = gerar_estado_sem_ruido(numero_passos, estado_inicial, modelo)
    trajetoria_com_ruido = gerar_estado_com_ruido(numero_passos, estado_inicial, modelo)

    eixo_tempo = np.arange(len(trajetoria_sem_ruido))

    return eixo_tempo, trajetoria_sem_ruido, trajetoria_com_ruido


def atualizar_frame(frame, ax, total_frames, delta_inicial=1e-6):

    ax.clear()

    if total_frames == 1:
        delta_atual = volatilidade
    else:
        delta_atual = delta_inicial + (volatilidade - delta_inicial) * frame / (total_frames - 1)

    eixo_tempo, trajetoria_sem_ruido, trajetoria_com_ruido = gerar_dados(delta_atual)

    ax.plot(eixo_tempo, trajetoria_sem_ruido, label="Sem ruído", linewidth=2)
    ax.plot(eixo_tempo, trajetoria_com_ruido, label="Com ruído de processo", linewidth=2, alpha=0.8)

    ax.set_title(f"Ruído de processo (δ = {delta_atual:.3f})")
    ax.set_xlabel("Passo de tempo (k)")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()


def animacao_ruido_processo(frames=30, interval=500, salvar=True):

    fig, ax = plt.subplots(figsize=(10, 5))

    anim = FuncAnimation(
        fig,
        atualizar_frame,
        frames=frames,
        interval=interval,
        fargs=(ax, frames)
    )

    if salvar:
        anim.save("animacoes/animacao_ruido_processo.mp4", writer="ffmpeg", fps=2)

    plt.show()
    plt.close(fig)