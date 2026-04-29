# Trabalho 1 - Teoria da Estimação

Este repositório contém a resolução do primeiro trabalho prático da disciplina de **Teoria da Estimação**. O projeto aborda a simulação de sistemas dinâmicos e a aplicação do **Método dos Mínimos Quadrados (MMQ)** para estimação de parâmetros e ajuste de dados.

---

## 📂 Estrutura do Projeto

- **`Part1/`**:  
  Simulação do modelo SIR e estimação dos parâmetros.

- **`Part2/`**:  
  Ajuste de curvas (linear, quadrático e restrito) para dados de renda e consumo.

- **`Projeto 1 - Teoria da Estimação.pdf`**:  
  Enunciado do trabalho.

- **`Relatorio_Projeto_1__Teoria_da_Estimacao.pdf`**:  
  Relatório final.

- **`Projeto 1 - Teoria da Estimação.pptx`**:  
  Apresentação do trabalho.

- **`README.md`**:  
  Visão geral do projeto.

---

## 📝 Descrição do Trabalho

O trabalho foi dividido em duas partes principais:

---

## 🔹 Parte 1 — Modelo SIR e Estimação de Parâmetros

Nesta etapa, foi estudado um modelo epidemiológico SIR discreto, descrito por:

$$
\begin{aligned}
S_{i+1} &= S_i - \beta S_i I_i, \\
I_{i+1} &= I_i + \beta S_i I_i - \gamma I_i, \\
R_{i+1} &= R_i + \gamma I_i,
\end{aligned}
$$

### Objetivos:

- Simular a evolução do sistema no intervalo \(i = 0, \dots, 350\);
- Estimar os parâmetros \(\beta\) (taxa de infecção) e \(\gamma\) (taxa de recuperação) via mínimos quadrados;
- Analisar o impacto da redução da amostragem nos parâmetros estimados.

### Configuração:

$$
S_0 = 0.9, \quad I_0 = 0.1, \quad R_0 = 0
$$

$$
\beta = 0.9, \quad \gamma = 0.1429 \approx \frac{1}{7}
$$

---

## 🔹 Parte 2 — Ajuste de Curvas e Regressão

Nesta etapa, foi analisada a relação entre **Renda (x)** e **Consumo (y)** a partir de dados observacionais.

### 📊 Dados Utilizados

| Renda (x) | Consumo (y) |
|----------|-------------|
| 139      | 122         |
| 126      | 114         |
| 90       | 86          |
| 144      | 134         |
| 163      | 146         |
| 136      | 107         |
| 61       | 68          |
| 62       | 117         |
| 41       | 71          |
| 120      | 98          |

---


### Procedimentos:

- Ajuste **linear** e **quadrático** via mínimos quadrados;
- Implementação de ajuste **com restrição de igualdade**, impondo que a curva passe pelo ponto:
  
  $$
  (163, 146)
  $$

- Comparação entre modelos com base em:
  - erro relativo;
  - RMSE (Root Mean Square Error);
  - análise gráfica.

---

## 📊 Resultados

- O modelo SIR apresentou comportamento consistente, com estabilização do sistema e parâmetros estimados próximos dos valores reais;
- A redução dos dados afetou a escala temporal dos parâmetros, mas preservou a dinâmica geral;
- No ajuste de curvas:
  - o modelo quadrático apresentou melhor desempenho global;
  - o modelo linear mostrou maior sensibilidade à imposição de restrições;
  - o ajuste restrito evidenciou o impacto de condições adicionais na modelagem.

---

## ⚙️ Ferramentas Utilizadas

- **Python 3.12**
- **NumPy** — operações matriciais e resolução de sistemas
- **Matplotlib** — geração de gráficos
- **Makefile** — automação do projeto

---

## 📌 Considerações

Este trabalho evidencia a versatilidade do **Método dos Mínimos Quadrados**, tanto na estimação de parâmetros em sistemas dinâmicos quanto no ajuste de modelos a dados reais, destacando sua importância em diversas áreas como epidemiologia, economia e engenharia.