# Trabalho 1 - Teoria da Estimação

Este repositório contém a resolução do primeiro trabalho prático da disciplina de **Teoria da Estimação**. O projeto aborda a simulação de sistemas dinâmicos e a aplicação do **Método dos Mínimos Quadrados (MMQ)** para estimativa de parâmetros e ajuste de dados.

---

## 📂 Estrutura do Projeto

- **`README.md`**: Visão geral do projeto.  
- **`resolucao/`**: Pasta contendo:
  - Código-fonte e scripts de execução.  
  - Relatório final em PDF/LaTeX.  

---

## 📝 Descrição do Trabalho

O trabalho foi dividido em duas partes principais:

### 1. Simulação e Estimação no Modelo SIR

O objetivo é analisar um **modelo epidemiológico SIR discreto**, definido pelas equações:

$$
\begin{aligned}
S_{i+1} &= S_i - \beta S_i I_i, \\
I_{i+1} &= I_i + \beta S_i I_i - g I_i, \\
R_{i+1} &= R_i + g I_i,
\end{aligned}
$$

**Objetivos:**

- Simular a evolução do modelo para o período \(i = 0, \dots, 350\).  
- Estimar os parâmetros \(\beta\) (taxa de infecção) e \(g\) (taxa de recuperação) utilizando o **Método dos Mínimos Quadrados**, a partir dos dados gerados na simulação.

**Condições iniciais utilizadas:**

$$
S_0 = 0.5, \quad I_0 = 0.1, \quad R_0 = 0, \quad \beta = 0.9, \quad g = 0.1429
$$

### 2. Ajuste de Curvas e Regressão

Nesta parte, a relação entre **Renda (X)** e **Consumo (Y)** é analisada com base em dados amostrais.

**Procedimentos realizados:**

- **Ajuste Linear e Quadrático**: aplicação do MMQ para determinar as curvas de melhor ajuste aos dados da tabela.  
- **Ajuste Restrito**: implementação do MMQ com restrição, garantindo que as curvas de ajuste passem exatamente pelo ponto \((163, 146)\).  

**Resultados Esperados:**

- Visualização das curvas ajustadas sobre os dados reais.  
- Comparação de erros relativos ponto a ponto e do **RMSE (Root Mean Square Error)** para cada modelo.  
- Análise do impacto da restrição na precisão dos ajustes.

---

Este projeto demonstra a utilidade do **MMQ** tanto na estimativa de parâmetros em sistemas dinâmicos quanto no ajuste de curvas a dados observacionais, com aplicações que vão desde epidemiologia até economia e engenharia.