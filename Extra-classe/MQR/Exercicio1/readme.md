# Exercício 6 — Mínimos Quadrados Regularizados

Durante um estudo de crescimento populacional de uma cultura de bactérias em laboratório, foram registradas medições da concentração de células ao longo do tempo.

Deseja-se modelar esse crescimento por um polinômio cúbico. Como modelos polinomiais de ordem mais elevada podem gerar instabilidades numéricas e parâmetros excessivamente elevados, deseja-se utilizar **regularização** para obter uma solução mais estável.

Considere que **todas as medições possuem a mesma confiabilidade**, ou seja, não será utilizada ponderação nas observações.

Os dados experimentais são apresentados abaixo:

| Tempo (h) | Concentração (milhões de células) |
| --------- | --------------------------------: |
| 0         |                               5.2 |
| 1         |                               8.1 |
| 2         |                              16.4 |
| 3         |                              31.8 |
| 4         |                              57.3 |
| 5         |                              96.7 |
| 6         |                             153.5 |

Deseja-se ajustar um modelo cúbico da forma:

[
C(t)=a+bt+ct^2+dt^3
]

Utilizando o **Método dos Mínimos Quadrados Regularizados**, determine os coeficientes:

[
a,\ b,\ c,\ d
]

considerando uma regularização da forma:

[
Q=\lambda I
]

com:

[
\lambda = 0.1
]

---

## Modelo matemático

A solução deve ser obtida utilizando:

[
\hat{x}=(Q+H^TH)^{-1}H^Ty
]

---

## Tarefas

1. Montar a matriz de regressão (H);
2. Construir a matriz de regularização (Q=\lambda I);
3. Estimar os parâmetros do modelo;
4. Plotar os pontos experimentais e a curva ajustada;
5. Calcular o NRMSE do ajuste;
6. Comparar os resultados com o mínimos quadrados tradicional (sem regularização).
