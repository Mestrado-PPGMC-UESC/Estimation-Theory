# Exercício 9 — Estimação de Correntes em Circuito Elétrico

## Enunciado

Em um experimento de laboratório de Engenharia Elétrica, deseja-se estimar a corrente elétrica que circula em três ramos de um circuito DC.

As correntes desconhecidas são:

- \(x_1\): corrente no ramo 1 (A)
- \(x_2\): corrente no ramo 2 (A)
- \(x_3\): corrente no ramo 3 (A)

Para realizar as medições, foram instalados sensores de corrente baseados em **resistores shunt** em cada ramo do circuito.

Um resistor shunt é um resistor de baixa resistência utilizado para medir corrente elétrica de forma indireta. Quando a corrente passa por esse resistor, surge uma queda de tensão em seus terminais.

Pela Lei de Ohm:

\[
V=RI
\]

cada sensor mede uma tensão proporcional à corrente elétrica que passa pelo respectivo ramo.

Além das medições de tensão, também estão disponíveis sensores de corrente em cada ramo. Para manter consistência física no processo de estimação, as correntes medidas foram convertidas para tensões equivalentes utilizando a Lei de Ohm.

---

## Sensores instalados

Os sensores shunt possuem as seguintes resistências:

- Ramo 1: \(R_1=1\Omega\)
- Ramo 2: \(R_2=2\Omega\)
- Ramo 3: \(R_3=3\Omega\)

---

## Medições experimentais

### Sensores de tensão

| Sensor | Tensão medida | Confiança |
|---|---:|---:|
| Ramo 1 | 2.10 V | 5 |
| Ramo 2 | 4.30 V | 4 |
| Ramo 3 | 5.85 V | 3 |

### Sensores de corrente

| Sensor | Corrente medida | Confiança |
|---|---:|---:|
| Ramo 1 | 2.05 A | 4 |
| Ramo 2 | 2.12 A | 4 |
| Ramo 3 | 1.90 A | 2 |

As correntes medidas devem ser convertidas em tensões equivalentes por:

\[
V=RI
\]

antes de serem utilizadas no vetor de medições.

---

## Restrição física

Pela Lei de Kirchhoff das Correntes, sabe-se que a soma das correntes que entram no nó do circuito deve ser:

\[
x_1+x_2+x_3=6A
\]

---

## Objetivo

Estimar os valores de:

\[
x_1,\ x_2,\ x_3
\]

utilizando o **Método dos Mínimos Quadrados Ponderados com Restrição**, resolvido pelo **método da função penalidade**.

---

## Análise

Resolver o problema para diferentes valores de:

\[
\mu \in \{0.1,\ 1,\ 10,\ 100,\ 1000\}
\]

e analisar:

- os valores estimados das correntes;
- a soma total das correntes;
- o erro de atendimento da restrição física.

---

## Comportamento esperado

À medida que o parâmetro de penalidade aumenta:

\[
\mu \uparrow
\]

espera-se que:

\[
x_1+x_2+x_3 \to 6A
\]

ou seja, a solução passa a obedecer cada vez mais a restrição imposta pelo circuito.