# Exercício 4 — Método dos Mínimos Quadrados Ponderados

Durante um experimento de calibração de temperatura de um reator químico, um sensor registrou a temperatura interna do sistema ao longo do tempo.

Sabe-se que cada medição possui um nível de confiança associado, devido a diferentes condições de aquisição dos dados.

Os dados coletados são apresentados abaixo:

| Tempo (s) | Temperatura (°C) | Confiança |
| --------- | ---------------- | --------- |
| 0         | 20.5             | 5         |
| 1         | 24.8             | 4         |
| 2         | 31.2             | 2         |
| 3         | 35.7             | 1         |
| 4         | 42.6             | 1         |

Considere que:

* valores maiores de confiança representam medições mais confiáveis;
* valores menores representam medições com maior incerteza.

Deseja-se modelar a temperatura por uma função linear do tipo:

[
T(t)=a+bt
]

Utilizando o **Método dos Mínimos Quadrados Ponderados**, determine os coeficientes:

[
a,\ b
]

que melhor ajustam os dados experimentais.
