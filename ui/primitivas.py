"""
Primitivas de desenho usadas pela mesa 3D e pela HUD 2D.

As funções daqui ficam deliberadamente simples: retângulos, linhas, círculos,
texto via textura e desenho básico de dominó. Módulos de nível mais alto cuidam
de layout, regra de jogo e interação.
"""

from OpenGL.GL import *
import math
import pygame


# ============================================================
# Cores básicas
# ============================================================

COR_MESA = (0.1, 0.5, 0.2)
COR_PECA = (0.95, 0.95, 0.95)
COR_LINHA = (0.1, 0.1, 0.1)


# ============================================================
# Primitivas OpenGL gerais
# ============================================================

def retangulo(x, y, w, h, cor, alpha=1.0):
    r, g, b = cor
    glColor4f(r, g, b, alpha)

    glBegin(GL_QUADS)
    glVertex2f(x,     y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x,     y + h)
    glEnd()


def contorno_retangulo(x, y, w, h, cor, alpha=1.0, largura=1.5):
    r, g, b = cor
    glColor4f(r, g, b, alpha)
    glLineWidth(largura)

    glBegin(GL_LINE_LOOP)
    glVertex2f(x,     y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x,     y + h)
    glEnd()


def linha(x1, y1, x2, y2, cor, alpha=1.0, largura=1.2):
    r, g, b = cor
    glColor4f(r, g, b, alpha)
    glLineWidth(largura)

    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def triangulo(x1, y1, x2, y2, x3, y3, cor, alpha=1.0):
    r, g, b = cor
    glColor4f(r, g, b, alpha)

    glBegin(GL_TRIANGLES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glVertex2f(x3, y3)
    glEnd()


def circulo(cx, cy, raio, cor, alpha=1.0, segmentos=18):
    r, g, b = cor
    glColor4f(r, g, b, alpha)

    glBegin(GL_POLYGON)

    for i in range(segmentos):
        theta = 2.0 * math.pi * i / segmentos
        x = cx + raio * math.cos(theta)
        y = cy + raio * math.sin(theta)
        glVertex2f(x, y)

    glEnd()


# ============================================================
# Texto em OpenGL usando pygame.font
# ============================================================

def _upload_texture(surface):
    # Cada texto vira uma textura temporária. Como a HUD é pequena, essa
    # abordagem é simples e suficiente para a visualização da aula.
    data = pygame.image.tostring(surface, "RGBA", True)
    w, h = surface.get_size()

    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        w,
        h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )

    return tid, w, h


def desenhar_texto(texto, fonte, cor, x, y, alpha=1.0):
    """
    Desenha texto em coordenadas de tela.

    x, y são o canto superior esquerdo do texto.
    Retorna (largura, altura).
    """
    surface = fonte.render(texto, True, cor).convert_alpha()
    tid, w, h = _upload_texture(surface)

    glEnable(GL_TEXTURE_2D)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glBindTexture(GL_TEXTURE_2D, tid)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(x,     y)

    glTexCoord2f(1, 1)
    glVertex2f(x + w, y)

    glTexCoord2f(1, 0)
    glVertex2f(x + w, y + h)

    glTexCoord2f(0, 0)
    glVertex2f(x,     y + h)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDeleteTextures(1, [tid])

    return w, h


# ============================================================
# Modo 2D para HUD
# ============================================================

def iniciar_2d(sw, sh):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    glOrtho(0, sw, sh, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def finalizar_2d():
    glDisable(GL_BLEND)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


# ============================================================
# Pontos de dominó
# ============================================================

def _posicoes_pontos(valor, ox, oy, y_para_baixo=False):
    """
    Devolve as posições relativas dos pontos de uma metade do dominó.

    Se y_para_baixo=True, usa coordenadas de tela.
    Se y_para_baixo=False, usa coordenadas matemáticas/OpenGL normais.
    """
    cima = -oy if y_para_baixo else oy
    baixo = oy if y_para_baixo else -oy

    posicoes = {
        0: [],
        1: [(0, 0)],
        2: [(-ox, cima), (ox, baixo)],
        3: [(-ox, cima), (0, 0), (ox, baixo)],
        4: [
            (-ox, cima),
            (ox, cima),
            (-ox, baixo),
            (ox, baixo),
        ],
        5: [
            (-ox, cima),
            (ox, cima),
            (0, 0),
            (-ox, baixo),
            (ox, baixo),
        ],
        6: [
            (-ox, cima),
            (ox, cima),
            (-ox, 0),
            (ox, 0),
            (-ox, baixo),
            (ox, baixo),
        ],
    }

    if valor not in posicoes:
        raise ValueError(f"Valor de dominó inválido: {valor}")

    return posicoes[valor]


# ============================================================
# Dominó da mesa: coordenadas locais OpenGL
# ============================================================

def desenhar_metade_domino(valor, centro_x, centro_y):
    raio_ponto = 0.12
    offset = 0.25

    for dx, dy in _posicoes_pontos(valor, offset, offset, y_para_baixo=False):
        circulo(
            centro_x + dx,
            centro_y + dy,
            raio_ponto,
            COR_LINHA,
        )


def desenhar_domino(valor_esq, valor_dir):
    """
    Desenha uma peça de dominó centrada na origem local.

    Tamanho local antes da escala:
        largura = 2.0
        altura  = 1.0
    """
    retangulo(-1.0, -0.5, 2.0, 1.0, COR_PECA)
    contorno_retangulo(-1.0, -0.5, 2.0, 1.0, COR_LINHA, largura=2.0)

    linha(0.0, -0.5, 0.0, 0.5, COR_LINHA, largura=2.0)

    desenhar_metade_domino(valor_esq, -0.5, 0.0)
    desenhar_metade_domino(valor_dir, 0.5, 0.0)


# ============================================================
# Dominó da HUD: coordenadas de tela
# ============================================================

def desenhar_verso_domino_2d(x, y, w, h, alpha=1.0):
    retangulo(x, y, w, h, (0.12, 0.16, 0.22), alpha)
    contorno_retangulo(x, y, w, h, (0.02, 0.02, 0.02), alpha, largura=1.5)

    retangulo(x + 4, y + 4, w - 8, h - 8, (0.20, 0.34, 0.48), alpha)

    linha(
        x + 7,
        y + h - 7,
        x + w - 7,
        y + 7,
        (0.65, 0.85, 1.0),
        alpha,
        largura=1.4,
    )


def desenhar_domino_2d(x, y, w, h, peca=None, verso=False, alpha=1.0):
    """
    Desenha dominó em coordenadas de tela.

    x, y são o canto superior esquerdo.
    w, h são largura e altura.
    Se verso=True, desenha o verso da peça.
    """
    if verso:
        desenhar_verso_domino_2d(x, y, w, h, alpha)
        return

    if not peca:
        return

    valor_esq, valor_dir = peca

    vertical = h > w

    half_w = w / 2.0 if vertical else w / 4.0
    half_h = h / 4.0 if vertical else h / 2.0
    raio = max(1.5, h * 0.095)

    retangulo(x, y, w, h, (0.94, 0.94, 0.92), alpha)
    contorno_retangulo(x, y, w, h, (0.02, 0.02, 0.02), alpha, largura=1.5)

    if vertical:
        meio = y + h / 2.0

        linha(x, meio, x + w, meio, (0.02, 0.02, 0.02), alpha, largura=1.1)

        centros = [
            (x + w * 0.5, y + h * 0.25),
            (x + w * 0.5, y + h * 0.75),
        ]

    else:
        meio = x + w / 2.0

        linha(meio, y, meio, y + h, (0.02, 0.02, 0.02), alpha, largura=1.1)

        centros = [
            (x + w * 0.25, y + h * 0.5),
            (x + w * 0.75, y + h * 0.5),
        ]

    for px, py in _pontos_2d(valor_esq, centros[0][0], centros[0][1], half_w, half_h):
        circulo(px, py, raio, (0.03, 0.03, 0.03), alpha, segmentos=14)

    for px, py in _pontos_2d(valor_dir, centros[1][0], centros[1][1], half_w, half_h):
        circulo(px, py, raio, (0.03, 0.03, 0.03), alpha, segmentos=14)


def _pontos_2d(valor, cx, cy, half_w, half_h):
    ox = half_w * 0.38
    oy = half_h * 0.45

    return [
        (cx + dx, cy + dy)
        for dx, dy in _posicoes_pontos(valor, ox, oy, y_para_baixo=True)
    ]
