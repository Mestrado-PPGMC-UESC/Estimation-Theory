# Exercício 9 — Mínimos Quadrados Ponderados com Restrição via Função Penalidade

## Contexto físico

Em um laboratório de Engenharia Elétrica, deseja-se estimar as correntes que circulam em três ramos de um circuito elétrico:

* (x_1): corrente no ramo 1 (A);
* (x_2): corrente no ramo 2 (A);
* (x_3): corrente no ramo 3 (A).

Por limitações físicas do sistema, não é possível medir diretamente cada corrente individualmente.

Em vez disso, sensores instalados em diferentes pontos do circuito medem **combinações lineares dessas correntes**.

---

## O que está sendo medido?

Cada sensor mede uma tensão proporcional à combinação das correntes do circuito.

As medições obtidas são:

$$
y_i
$$

onde:

$$
y_i = \text{tensão medida pelo sensor } i
$$

Unidade:

$$
\text{Volts (V)}
$$

---

## Modelo físico

As medições obedecem:

$$
y \approx Hx
$$

onde:

$$
x=
\begin{bmatrix}
x_1\
x_2\
x_3
\end{bmatrix}
$$

representa as correntes desconhecidas.

---

## Dados experimentais

| Sensor | Ganho para (x_1) | Ganho para (x_2) | Ganho para (x_3) | Tensão medida (V) | Confiança |
| ------ | ---------------: | ---------------: | ---------------: | ----------------: | --------: |
| 1      |              1.0 |              0.5 |              0.2 |              2.62 |         5 |
| 2      |              0.8 |              0.7 |              0.3 |              2.74 |         4 |
| 3      |              0.6 |              1.0 |              0.5 |              2.91 |         3 |
| 4      |              0.4 |              1.2 |              0.8 |              3.05 |         2 |
| 5      |              0.3 |              1.5 |              1.0 |              3.24 |         1 |

---

## Interpretação da matriz (H)

Cada linha representa a sensibilidade de um sensor às correntes do circuito.

Exemplo no sensor 1:

$$
y_1 = 1.0x_1 + 0.5x_2 + 0.2x_3
$$

ou seja:

* sensor 1 é mais sensível à corrente do ramo 1;
* possui sensibilidade intermediária ao ramo 2;
* possui baixa sensibilidade ao ramo 3.

---

## Restrição física

Pela Lei de Kirchhoff das Correntes, sabe-se que:

$$
x_1+x_2+x_3=6A
$$

ou seja:

a corrente total que entra no nó do circuito é de:

$$
6A
$$

---

## Objetivo

Estimar:

$$
x_1,\ x_2,\ x_3
$$

utilizando o **Método dos Mínimos Quadrados Ponderados com Restrição**, resolvido via **função penalidade**.

---

## Tarefas

1. Montar a matriz (H);
2. Montar o vetor de medições (y);
3. Construir a matriz de ponderação (W);
4. Definir a restrição:

$$
G=
\begin{bmatrix}
1 & 1 & 1
\end{bmatrix}
$$

$$
u=
\begin{bmatrix}
6
\end{bmatrix}
$$

5. Resolver para diferentes valores de:

$$
\mu \in {0.1,\ 1,\ 10,\ 100,\ 1000}
$$

6. Observar o comportamento das correntes estimadas e da restrição.
