# Exercício — Estimação de Consumo de Combustível em Drones

## Enunciado

Uma equipe de engenharia está monitorando o consumo instantâneo de combustível de três drones autônomos durante uma missão de reconhecimento.

Os sensores embarcados forneceram as seguintes medições de consumo:

| Drone | Consumo medido (L/h) |
|-------|----------------------|
| Drone 1 | 3.20 |
| Drone 2 | 2.80 |
| Drone 3 | 3.50 |

Cada sensor possui um nível de confiabilidade diferente:

| Sensor | Peso de confiança |
|--------|-------------------|
| Sensor 1 | 5 |
| Sensor 2 | 4 |
| Sensor 3 | 3 |

Além disso, pelo planejamento operacional da missão, sabe-se que o consumo total dos três drones deve ser:

\[
x_1 + x_2 + x_3 = 10.0 \text{ L/h}
\]

onde:

- \(x_1\): consumo real do drone 1  
- \(x_2\): consumo real do drone 2  
- \(x_3\): consumo real do drone 3  

---

## Objetivo

Estimar os consumos reais dos três drones utilizando:

- **Mínimos Quadrados Ponderados**
- **Restrição física**
- **Regularização com \(\mu\)**

---

