# Parte 1: Simulação e Estimação - Modelo SIR

Este diretório contém a implementação da primeira questão do trabalho, focada na **simulação do modelo epidemiológico SIR** e na análise da sensibilidade do **estimador via Mínimos Quadrados (MMQ)**.

---

## 📂 Estrutura de Arquivos

- **`main.py`**: Coordena a simulação completa e as duas etapas de estimação.  
- **`model.py`**: Contém as equações de diferença do modelo SIR.  
- **`estimacao.py`**: Implementa o **Método dos Mínimos Quadrados (MMQ)** para estimar \(\beta\) e \(g\).  
- **`plotagem.py`**: Gera curvas de \(S_i, I_i, R_i\) e comparações entre estimações.  
- **`config.py`**: Armazena os parâmetros do modelo:
  $$
  \beta = 0.9, \quad g = 0.1429
  $$
  e condições iniciais:
  $$
  S_0 = 0.5, \quad I_0 = 0.1, \quad R_0 = 0
  $$
- **`requirements.txt`**: Dependências do projeto.

---

## 📝 Objetivos da Estimação

O projeto realiza a identificação de parâmetros em dois cenários distintos, com o objetivo de avaliar a **precisão do modelo e a robustez do estimador**:

1. **Estimação Completa:**  
   Utiliza o conjunto total de dados simulados (\(L = 350\)).

2. **Estimação com Dados Parciais:**  
   Utiliza apenas a metade inicial dos dados (\(L = 175\)), para observar o comportamento do algoritmo com uma **janela de observação reduzida**, especialmente antes ou durante o pico de infecção.

---

## 🚀 Como Executar

A execução é simplificada via **Makefile**. Siga os passos abaixo:

### 1. Instalação

Crie o ambiente necessário e instale as bibliotecas:

```bash
make install
```
### 2. Execução

Roda a simulação SIR, executa as duas baterias de estimação (dados totais vs. parciais) e exibe os resultados:

```bash
make run
```
