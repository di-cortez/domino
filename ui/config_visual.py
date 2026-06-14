"""
Constantes de geometria visual da mesa.

Alterar estes valores muda o tamanho das peças, o espaçamento e os limites que
o algoritmo de layout usa para decidir quando virar um ramo.
"""

POS_X_INICIAL = 0.0
POS_Y_INICIAL = 3.35

ESCALA_PECA = 0.75
GAP_ENTRE_PECAS = 0.10

LIMITE_X = 8.0

LARGURA_PECA_DEITADA = 2.0 * ESCALA_PECA
ALTURA_PECA_DEITADA = 1.0 * ESCALA_PECA

LARGURA_PECA_EM_PE = 1.0 * ESCALA_PECA
ALTURA_PECA_EM_PE = 2.0 * ESCALA_PECA

LIMITE_Y_INFERIOR = -5.0
LIMITE_Y_SUPERIOR = POS_Y_INICIAL + ALTURA_PECA_EM_PE / 2.0
