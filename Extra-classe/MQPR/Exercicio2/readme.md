# Exercício 10 — Estimação de Potências em Cargas Elétricas

## Enunciado

Em um laboratório de Engenharia Elétrica, deseja-se estimar a potência dissipada em três cargas resistivas ligadas em paralelo a uma fonte DC.

As potências desconhecidas são:

* (x_1): potência dissipada na carga 1 (W)
* (x_2): potência dissipada na carga 2 (W)
* (x_3): potência dissipada na carga 3 (W)

Para monitorar o sistema, sensores de corrente foram instalados em cada ramo.

Como a tensão de alimentação é constante, igual a:

[
V=12V
]

a potência em cada carga pode ser modelada por:

[
P=VI
]

onde:

* (P): potência elétrica (W)
* (V): tensão da fonte (V)
* (I): corrente elétrica (A)

---

## Correntes medidas experimentalmente

| Carga   | Corrente medida | Confiança |
| ------- | --------------: | --------: |
| Carga 1 |          2.10 A |         5 |
| Carga 2 |          1.85 A |         4 |
| Carga 3 |          2.35 A |         3 |

---

## Restrição física

Sabe-se, a partir das especificações da fonte, que a potência total entregue ao circuito deve ser:

[
x_1+x_2+x_3=75W
]

---

## Objetivo

Estimar:

[
x_1,\ x_2,\ x_3
]

utilizando o **Método dos Mínimos Quadrados Ponderados com Restrição**, resolvido pelo **método da função penalidade**.

---

## Análise

Resolver o problema para diferentes valores de:

[
\mu \in {0.1,\ 1,\ 10,\ 100,\ 1000}
]

e analisar:

* as potências estimadas;
* a soma total das potências;
* o erro de atendimento da restrição física.
