
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from model import ModeloPreco
from simulacao import gerar_estado_sem_ruido, gerar_estado_com_ruido, gerar_observacao
from kalman import estimativa_filtrada
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

def gerar_dados(delta_atual, q_atual):

    np.random.seed(semente_aleatoria)

    modelo = ModeloPreco(
            taxa_crescimento,
            volatilidade,
            passo_tempo,
            ruido_observacao,
            tendencia_sistematica
        )


    p_sem_ruido = gerar_estado_sem_ruido(numero_passos, estado_inicial, modelo)
    p_com_ruido = gerar_estado_com_ruido(numero_passos, estado_inicial, modelo)
    y = gerar_observacao(p_com_ruido, modelo)

    k = np.arange(len(p_sem_ruido))

    return k, p_sem_ruido, p_com_ruido, y


def atualizar_delta(frame, ax, frames, delta_inicial=1e-6):

    ax.clear()

    delta_atual = delta_inicial + (delta - delta_inicial) * frame / (frames - 1)
    q_atual = 0.0

    k, p_sem_ruido, p_com_ruido, y = gerar_dados(delta_atual, q_atual)

    ax.plot(k, p_sem_ruido, label="Sem ruído", linewidth=2)
    ax.plot(k, p_com_ruido, label="Com ruído de processo", linewidth=2, alpha=0.8)

    ax.set_title(f"Ruído de processo | delta = {delta_atual:.3f} | q = {q_atual:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()


def atualizar_q(frame, ax, frames, q_inicial=1e-6):


    ax.clear()

    delta_atual = 0.0
    q_atual = q_inicial + (q - q_inicial) * frame / (frames - 1)

    k, p_sem_ruido, p_com_ruido, y = gerar_dados(delta_atual, q_atual)

    ax.plot(k, p_sem_ruido, label="Sem ruído", linewidth=2)
    ax.scatter(k, y, label="Observação", s=12, alpha=0.6)

    ax.set_title(f"Ruído de observação | delta = {delta_atual:.3f} | q = {q_atual:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()


def atualizar_ambos(frame, ax, frames, delta_inicial=1e-6, q_inicial=1e-6):

    ax.clear()

    delta_atual = delta_inicial + (delta - delta_inicial) * frame / (frames - 1)
    q_atual = q_inicial + (q - q_inicial) * frame / (frames - 1)

    k, p_sem_ruido, p_com_ruido, y = gerar_dados(delta_atual, q_atual)

    ax.plot(k, p_sem_ruido, label="Sem ruído", linewidth=2)
    ax.plot(k, p_com_ruido, label="Com ruído de processo", linewidth=2, alpha=0.8)
    ax.scatter(k, y, label="Observação", s=12, alpha=0.6)

    ax.set_title(f"Ruídos juntos | delta = {delta_atual:.3f} | q = {q_atual:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()


def animacao_ruido_processo(frames=30, interval=500, salvar=True):

    fig, ax = plt.subplots(figsize=(10, 5))

    anim = FuncAnimation(
        fig,
        atualizar_delta,
        frames=frames,
        interval=interval,
        fargs=(ax, frames)
    )

    if salvar:
        anim.save("animacoes/animacao_ruido_processo.mp4", writer="ffmpeg", fps=2)

    plt.show()


def animacao_ruido_observacao(frames=30, interval=500, salvar=True):


    fig, ax = plt.subplots(figsize=(10, 5))

    anim = FuncAnimation(
        fig,
        atualizar_q,
        frames=frames,
        interval=interval,
        fargs=(ax, frames)
    )

    if salvar:
        anim.save("animacoes/animacao_ruido_observacao.mp4", writer="ffmpeg", fps=2)

    plt.show()


def animacao_ruidos_juntos(frames=30, interval=500, salvar=True):


    fig, ax = plt.subplots(figsize=(10, 5))

    anim = FuncAnimation(
        fig,
        atualizar_ambos,
        frames=frames,
        interval=interval,
        fargs=(ax, frames)
    )

    if salvar:
        anim.save("animacoes/animacao_ruidos_juntos.mp4", writer="ffmpeg", fps=2)

    plt.show()


def atualizar_Q(frame, ax, frames, Q_inicial=1e-6):

    ax.clear()

    Q_atual = Q_inicial + (delta**2 - Q_inicial) * frame / (frames - 1)
    R_atual = q**2

    k, p_sem_ruido, p_com_ruido, y = gerar_dados(delta, q)

    x_filt, _, _ = estimativa_filtrada(y, Q_atual, R_atual, x0=p0, P0=1.0)

    ax.plot(k, p_com_ruido, label="Estado real", linewidth=2)
    ax.scatter(k, y, label="Observação", s=12, alpha=0.6)
    ax.plot(k, x_filt, label="Filtrado", linestyle="--", linewidth=2)

    ax.set_title(f"Variação de Q | Q = {Q_atual:.4f} | R = {R_atual:.4f}")
    ax.set_xlabel("k")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()

def animacao_variando_Q(frames=30, interval=500, salvar=True):


    fig, ax = plt.subplots(figsize=(10, 5))

    anim = FuncAnimation(
        fig,
        atualizar_Q,
        frames=frames,
        interval=interval,
        fargs=(ax, frames)
    )

    if salvar:
        anim.save("animacoes/animacao_Q.mp4", writer="ffmpeg", fps=2)

    plt.show()

def atualizar_R(frame, ax, frames, R_inicial=1e-6):


    ax.clear()

    Q_atual = delta**2
    R_atual = R_inicial + (q**2 - R_inicial) * frame / (frames - 1)

    k, p_sem_ruido, p_com_ruido, y = gerar_dados(delta, q)

    x_filt, _, _ = estimativa_filtrada(y, Q_atual, R_atual, x0=p0, P0=1.0)

    ax.plot(k, p_com_ruido, label="Estado real", linewidth=2)
    ax.scatter(k, y, label="Observação", s=12, alpha=0.6)
    ax.plot(k, x_filt, label="Filtrado", linestyle="--", linewidth=2)

    ax.set_title(f"Variação de R | Q = {Q_atual:.4f} | R = {R_atual:.4f}")
    ax.set_xlabel("k")
    ax.set_ylabel("Preço")
    ax.grid(True)
    ax.legend()

