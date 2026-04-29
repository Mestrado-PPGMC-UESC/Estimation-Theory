from config import renda, consumo
from ajustes import ajustar_linear,ajustar_quadratico,ajustar_linear_restrito,ajustar_quadratico_restrito,ajustar_cubico
from plotagem import plotar_ajustes
from auxiliar import comparacao

a1, b1 = ajustar_linear(renda,consumo)
a2, b2, c2 = ajustar_quadratico(renda,consumo)
a3,b3,c3,d3 = ajustar_ajustar_cubico(renda,consumo)

print('Ajuste Linear:')
print(f'y = {a1:.4f}x + {b1:.4f}')

print()

print('Ajuste Quadrático:')
print(f'y = {a2:.4f}x² + {b2:.4f}x + {c2:.4f}')

print('Ajuste Cubico:')
print(f'y ={a4:.4f}x³ + {a3:.4f}x² + {b3:.4f}x + {c3:.4f}')

plotar_ajustes(renda, consumo, a1, b1, a2, b2, c2,'Ajuste Linear e Quadrático')



comparacao(renda,consumo,a1,b1,a2,b2,c2)

# ----------------------------------------------


a1_r, b1_r = ajustar_linear_restrito(renda,consumo)
a2_r, b2_r, c2_r = ajustar_quadratico_restrito(renda,consumo)

print('Ajuste Linear Restrito:')
print(f'y = {a1_r:.4f}x + {b1_r:.4f}')

print('Ajuste Quadrático Restrito:')
print(f'y = {a2_r:.4f}x² + {b2_r:.4f}x + {c2_r:.4f}')

plotar_ajustes(renda, consumo, a1_r, b1_r, a2_r, b2_r, c2_r,'Ajuste Linear e Quadrático Restrito','Restrito')

# Comparação dos ajustes restritos
print('Comparação - Ajustes Restritos')
comparacao(renda, consumo, a1_r, b1_r, a2_r, b2_r, c2_r)