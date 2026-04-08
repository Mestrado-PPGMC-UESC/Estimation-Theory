from config import beta,gamma, S0, I0 , R0 , n_passos
from model import simular_sir
from plotagem import plotar_simulacao,plotar_comparacao,plotar_comparacao_2
from estimacao import estimar_parametros


S, I, R = simular_sir(beta, gamma, S0, I0, R0, n_passos)

plotar_simulacao(S, I, R)


# Estimação dos parâmetros
beta_estimado, gamma_estimado = estimar_parametros(S, I, R)

# Resultados
print(f'Beta real     = {beta}')
print(f'Beta estimado = {beta_estimado:.16f}')
print(f'Gamma real     = {gamma}')
print(f'Gamma estimado = {gamma_estimado:.16f}')


S_estimado, I_estimado, R_estimado = simular_sir(beta_estimado, gamma_estimado, S0, I0, R0, n_passos)

plotar_comparacao(S, I, R, S_estimado, I_estimado, R_estimado)

# ==========================================
# Estimação usando apenas metade dos dados
# (índices 0, 2, 4, 6, ...)
# ==========================================

S_reduzido = S[::2]
I_reduzido = I[::2]
R_reduzido = R[::2]

beta_estimado_reduzido, gamma_estimado_reduzido = estimar_parametros(S_reduzido,I_reduzido,R_reduzido)

print('Usando apenas metade dos dados:')
print(f'Beta estimado reduzido  = {beta_estimado_reduzido:.4f}')
print(f'Gamma estimado reduzido = {gamma_estimado_reduzido:.4f}')

# Simulação usando os parâmetros estimados com metade dos dados
S_estimado_reduzido, I_estimado_reduzido, R_estimado_reduzido = simular_sir(beta_estimado_reduzido/2,gamma_estimado_reduzido/2,S0,I0,R0,n_passos)

# Comparação das curvas reconstruídas
plotar_comparacao_2(S,I,R,S_estimado, I_estimado, R_estimado,S_estimado_reduzido,I_estimado_reduzido,R_estimado_reduzido)