# Exercício 5 — Mínimos Quadrados Ponderados

## Enunciado

Durante testes de descarga de uma bateria utilizada em um sistema embarcado, foram realizadas medições da tensão elétrica ao longo do tempo.

Sabe-se que algumas medições foram obtidas com instrumentos mais precisos, enquanto outras sofreram interferências externas, resultando em diferentes níveis de confiança.

Os dados experimentais são apresentados abaixo:

| Tempo (min) | Tensão (V) | Confiança |
|---|---:|---:|
| 0 | 12.8 | 5 |
| 2 | 12.1 | 4 |
| 4 | 11.5 | 3 |
| 6 | 10.6 | 2 |
| 8 | 9.8 | 1 |

Deseja-se modelar a variação da tensão por uma função linear do tipo:

\[
V(t)=a+bt
\]

Utilizando o **Método dos Mínimos Quadrados Ponderados**, determine os coeficientes:

\[
a,\ b
\]

que melhor ajustam os dados experimentais.

---

## Objetivo

Implementar o método dos mínimos quadrados ponderados para estimar os parâmetros do modelo linear, considerando diferentes níveis de confiança para cada medição.

---