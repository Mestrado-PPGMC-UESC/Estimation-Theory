# Exercício 8 — Mínimos Quadrados Regularizados e Ponderados

## Enunciado

Durante um estudo de crescimento populacional de uma cultura de bactérias em laboratório, foram registradas medições da concentração de células ao longo do tempo.

Sabe-se que algumas medições foram realizadas em condições ideais, enquanto outras sofreram interferências experimentais, resultando em diferentes níveis de confiança.

Além disso, como o modelo adotado possui múltiplos parâmetros e pode apresentar instabilidades numéricas, deseja-se utilizar **regularização** para obter uma solução mais estável.

Os dados experimentais são apresentados abaixo:

| Tempo (h) | Concentração (milhões de células) | Confiança |
|---|---:|---:|
| 0 | 5.1 | 5 |
| 1 | 10.8 | 4 |
| 2 | 13.2 | 1 |
| 3 | 39.7 | 3 |
| 4 | 48.4 | 1 |
| 5 | 112.6 | 2 |
| 6 | 141.3 | 1 |

Deseja-se ajustar um modelo cúbico da forma:

\[
C(t)=a+bt+ct^2+dt^3
\]

Utilizando o **Método dos Mínimos Quadrados Regularizados e Ponderados**, determine os coeficientes:

\[
a,\ b,\ c,\ d
\]

considerando:

## Matriz de ponderação

A matriz de ponderação deve ser construída a partir dos níveis de confiança informados:

\[
w=[5,\ 4,\ 1,\ 3,\ 1,\ 2,\ 1]
\]

## Matriz de regularização

Considere:

\[
Q=\lambda I
\]

com:

\[
\lambda = 0.1
\]

---

## Modelo matemático

A solução deve ser obtida utilizando:

\[
\hat{x}=(Q+H^TWH)^{-1}H^TWy
\]

---

## Tarefas

1. Montar a matriz de regressão \(H\);
2. Construir a matriz de ponderação \(W\);
3. Construir a matriz de regularização \(Q\);
4. Estimar os parâmetros do modelo;
5. Plotar os pontos experimentais e a curva ajustada;
6. Calcular o NRMSE do ajuste;
7. Comparar os resultados com:
   - mínimos quadrados comum;
   - mínimos quadrados ponderado;
   - mínimos quadrados regularizado.