"""
Desenho 3D de uma peça individual.

As posições e ângulos são calculados em `layout_domino.py`; aqui apenas
aplicamos transformações OpenGL e chamamos a primitiva de dominó.
"""

from OpenGL.GL import *

from ui.primitivas import desenhar_domino
from ui.layout_domino import angulo_em_linha
from ui.config_visual import ESCALA_PECA


def desenhar_peca(info, pos_x, pos_y, angulo=None, valores=None):
    if valores is None:
        valores = tuple(info["peca"])

    if angulo is None:
        angulo = angulo_em_linha(info)

    valor_esq, valor_dir = valores

    glPushMatrix()

    glTranslatef(pos_x, pos_y, 0.0)
    glScalef(ESCALA_PECA, ESCALA_PECA, 1.0)

    if angulo != 0.0:
        glRotatef(angulo, 0.0, 0.0, 1.0)

    desenhar_domino(valor_esq, valor_dir)

    glPopMatrix()
