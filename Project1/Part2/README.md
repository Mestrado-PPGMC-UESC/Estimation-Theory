# Parte 2: Ajuste de Curvas e Regressão (MMQ)

Este diretório contém a resolução da segunda questão do trabalho, que analisa a relação entre **Renda (X)** e **Consumo (Y)** por meio de técnicas de **regressão estatística** e ajuste de curvas via **Método dos Mínimos Quadrados (MMQ)**.

---

## 📂 Estrutura de Arquivos

- **`main.py`**: Executa o fluxo principal de análise de dados e gera as saídas comparativas.  
- **`ajustes.py`**: Implementa o MMQ para os modelos:
  - Linear
  - Quadrático
  - Restrito (forçando a passagem pelo ponto de controle)
- **`plotagem.py`**: Gera gráficos de dispersão e as curvas de tendência resultantes.  
- **`config.py`**: Contém o dataset (tabela de Renda/Consumo) e as coordenadas do ponto de restrição $(x_r, y_r) = (163, 146)$.  
- **`auxiliar.py`**: Funções de suporte para manipulação de matrizes e cálculos matemáticos.  
- **`requirements.txt`**: Dependências necessárias para execução.

---

## 📝 Objetivos da Análise

O objetivo é modelar o **comportamento do consumo em função da renda** utilizando três abordagens de estimação:

1. **Ajuste Linear:**  
   Determinação da reta de tendência simples:
   $$
   y = a x + b
   $$

2. **Ajuste Quadrático:**  
   Modelagem parabólica para capturar variações não-lineares:
   $$
   y = a x^2 + b x + c
   $$

3. **Ajuste com Restrição:**  
   Aplicação do **MMQ Restrito** para garantir que os modelos passem obrigatoriamente pelo ponto de controle:
   $$
   (x_r, y_r) = (163, 146)
   $$

Estas abordagens permitem comparar a **flexibilidade do modelo** e avaliar o impacto de **condições de contorno conhecidas** na qualidade do ajuste.

---

## 🚀 Como Executar

A execução é padronizada via **Makefile**. Siga os passos:

### 1. Instalação

Instale as bibliotecas de álgebra linear e visualização:

```bash
make install
```

### 2. Execução

Roda a simulação SIR, executa as duas baterias de estimação (dados totais vs. parciais) e exibe os resultados:

```bash
make run
```
