"""
Renderização 3D da mesa.

Recebe um snapshot visual produzido pelo controlador, calcula um pivô estável
para a cadeia de dominós e desenha os dois ramos a partir desse pivô.
"""

from OpenGL.GL import *

from ui.primitivas import COR_MESA, retangulo
from ui.renderizador_estado import RenderizadorEstado
from ui.config_visual import POS_X_INICIAL, POS_Y_INICIAL
from ui.layout_domino import (
    angulo_em_linha,
    quebrar_cadeia_no_pivo,
    calcular_slots_ramo,
)
from ui.desenho_domino import desenhar_peca

_renderizador = RenderizadorEstado()


def desenhar_mesa():
    retangulo(-20.0, -20.0, 40.0, 40.0, COR_MESA)


def renderizar_cena(estado):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -15.0)

    desenhar_mesa()

    cadeia_visual = estado.get("cadeia_visual", [])

    if not cadeia_visual:
        return

    # O pivô evita que toda a cadeia deslize quando uma peça entra em uma das
    # pontas. A mesa cresce para os dois lados a partir desse ponto fixo.
    indice_pivo = _renderizador.obter_indice_pivo(cadeia_visual)

    if indice_pivo is None:
        return

    lado_esquerdo, pivo, lado_direito = quebrar_cadeia_no_pivo(
        cadeia_visual,
        indice_pivo,
    )

    # Primeiro desenha o pivô; depois cada lado é calculado como um ramo
    # independente, permitindo curvas sem deslocar o centro da mesa.
    desenhar_peca(
    pivo,
    POS_X_INICIAL,
    POS_Y_INICIAL,
    angulo=angulo_em_linha(pivo),
    valores=tuple(pivo["peca"]),
)

    slots_esquerda = calcular_slots_ramo(
        lado_esquerdo,
        pivo,
        direcao=-1,
        pos_x_inicial=POS_X_INICIAL,
        pos_y_inicial=POS_Y_INICIAL,
    )

    slots_direita = calcular_slots_ramo(
        lado_direito,
        pivo,
        direcao=1,
        pos_x_inicial=POS_X_INICIAL,
        pos_y_inicial=POS_Y_INICIAL,
    )

    for info, slot in slots_esquerda:
        desenhar_peca(info, slot["pos_x"], slot["pos_y"], slot["angulo"], slot["valores"])

    for info, slot in slots_direita:
        desenhar_peca(info, slot["pos_x"], slot["pos_y"], slot["angulo"], slot["valores"])
