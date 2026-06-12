from OpenGL.GL import *
from OpenGL.GLU import *
import math
import pygame

class RenderizadorEspacial:
    def __init__(self, limite_x=8.0, gap_x=0.1, passo_y=0.3, pos_y_inicial=4.5):
        self.MAX_X = limite_x
        self.gap_x = gap_x
        self.passo_y = passo_y
        self.pos_y_inicial = pos_y_inicial

    def _obter_angulo(self, info):
        return 90.0 if info.get("orientacao") == "vertical" else 0.0

    def _obter_raio_x(self, info):
        """Returns the horizontal half-width of the piece."""
        return 0.5 if info.get("orientacao") == "vertical" else 1.0

    def _obter_altura_y(self, info):
        """Returns the vertical footprint of the piece."""
        return 2.0 if info.get("orientacao") == "vertical" else 1.0

    def _calcular_passo_x(self, info_anterior, info_atual):
        raio_x_ant = self._obter_raio_x(info_anterior)
        raio_x_atu = self._obter_raio_x(info_atual)
        return raio_x_ant + raio_x_atu + self.gap_x

    def _clamp_x(self, x, raio):
        """Keep a piece's center so that its full width stays inside [-MAX_X, MAX_X]."""
        limite = self.MAX_X - raio
        return max(-limite, min(limite, x))

    def _raio_render(self, info):
        """Meia-largura em X conforme a orientação JÁ resolvida para render."""
        return 0.5 if info.get("orient_render") == "vertical" else 1.0
 
 
    def _altura_render(self, info):
        """Altura (footprint em Y) conforme a orientação JÁ resolvida para render."""
        return 2.0 if info.get("orient_render") == "vertical" else 1.0
 
 
    def calcular_layout(self, cadeia_visual):
        if not cadeia_visual:
            return []
    
        pecas = [info.copy() for info in cadeia_visual]
    
        # ---------- Passo 1: X, orientação de render, linhas e cotovelos ----------
        linhas = []        # cada item: (lista_de_pecas, direcao_da_linha)
        cotovelos = []     # cotovelo[k] liga a linha k à linha k+1 (peça de canto vertical)
        linha_atual = []
        direcao = 1        # +1 = esquerda -> direita ; -1 = direita -> esquerda
        pos_x = None
        info_anterior = None
    
        for info in pecas:
            # Duplas já chegam como "vertical" (atravessadas); o resto é horizontal.
            natural_vertical = info.get("orientacao") == "vertical"
            info["cotovelo"] = False
            info["dir_linha"] = direcao
            info["orient_render"] = "vertical" if natural_vertical else "horizontal"
    
            if info_anterior is None:
                # Primeira peça encosta na borda esquerda para usar a largura toda.
                pos_x = -self.MAX_X + self._raio_render(info)
                info["pos_x"] = pos_x
                linha_atual.append(info)
                info_anterior = info
                continue
    
            # Tenta encaixar na linha com a orientação natural.
            passo_x = self._raio_render(info_anterior) + self._raio_render(info) + self.gap_x
            candidato_x = pos_x + passo_x * direcao
            raio = self._raio_render(info)
            estoura = (direcao == 1 and candidato_x + raio > self.MAX_X) or \
                    (direcao == -1 and candidato_x - raio < -self.MAX_X)
    
            if estoura:
                # Vira COTOVELO: girada na vertical, no X da peça terminal, fechando
                # a linha. A próxima peça conecta a ele e segue no sentido oposto.
                info["orient_render"] = "vertical"
                info["cotovelo"] = True
                info["pos_x"] = info_anterior["pos_x"]
                linhas.append((linha_atual, direcao))
                cotovelos.append(info)
                linha_atual = []
                direcao *= -1
                pos_x = info["pos_x"]
                info_anterior = info          # nova linha parte do cotovelo
            else:
                pos_x = candidato_x
                info["pos_x"] = pos_x
                linha_atual.append(info)
                info_anterior = info
    
        if linha_atual:
            linhas.append((linha_atual, direcao))
    
        # ---------- Passo 2: Y, espaçando pela altura real + o cotovelo entre linhas ----------
        layout = []
        gap = self.passo_y
        y_linha = self.pos_y_inicial
        h_prev = None
    
        for k, (linha, dir_linha) in enumerate(linhas):
            h_linha = max((self._altura_render(p) for p in linha), default=1.0)
    
            if h_prev is not None:
                # meia linha anterior + folga + cotovelo (2.0) + folga + meia linha atual
                y_linha = y_linha - (h_prev / 2.0 + gap + 2.0 + gap + h_linha / 2.0)
    
            for p in linha:
                p["pos_y"] = y_linha
                p["angulo"] = 90.0 if p["orient_render"] == "vertical" else 0.0
                # 'dobrado' = linha invertida -> renderizador soma 180° para os pips
                # continuarem encaixando no sentido certo.
                p["dobrado"] = (dir_linha == -1)
                layout.append(p)
    
            # Cotovelo que SAI desta linha rumo à próxima (centrado na folga abaixo).
            if k < len(cotovelos):
                cot = cotovelos[k]
                cot["pos_y"] = y_linha - (h_linha / 2.0 + gap + 1.0)
                cot["angulo"] = -90.0
                cot["dobrado"] = False
                layout.append(cot)
    
            h_prev = h_linha
    
        return layout


def desenhar_circulo(cx, cy, raio, num_segmentos=15):
    glBegin(GL_POLYGON)
    for i in range(num_segmentos):
        theta = 2.0 * math.pi * float(i) / float(num_segmentos)
        x = raio * math.cos(theta)
        y = raio * math.sin(theta)
        glVertex2f(x + cx, y + cy)
    glEnd()

def desenhar_metade_domino(valor, centro_x, centro_y):
    glColor3f(0.1, 0.1, 0.1) 
    raio_ponto = 0.12
    offset = 0.25 

    # Issue 3 Fix: Definição explícita para o valor 0 (branco) para evitar falhas silenciosas
    posicoes = {
        0: [], 
        1: [(0, 0)],
        2: [(-offset, offset), (offset, -offset)],
        3: [(-offset, offset), (0, 0), (offset, -offset)],
        4: [(-offset, offset), (offset, offset), (-offset, -offset), (offset, -offset)],
        5: [(-offset, offset), (offset, offset), (0, 0), (-offset, -offset), (offset, -offset)],
        6: [(-offset, offset), (offset, offset), (-offset, 0), (offset, 0), (-offset, -offset), (offset, -offset)]
    }

    if valor not in posicoes:
        raise ValueError(f"Valor de dominó inválido passado ao renderizador: {valor}")

    for dx, dy in posicoes[valor]:
        desenhar_circulo(centro_x + dx, centro_y + dy, raio_ponto)

def desenhar_domino(valor_esq, valor_dir):
    glColor3f(0.95, 0.95, 0.95)
    glBegin(GL_QUADS)
    glVertex2f(-1.0, -0.5)
    glVertex2f( 1.0, -0.5)
    glVertex2f( 1.0,  0.5)
    glVertex2f(-1.0,  0.5)
    glEnd()

    glColor3f(0.1, 0.1, 0.1)
    glLineWidth(2.0)
    
    glBegin(GL_LINE_LOOP)
    glVertex2f(-1.0, -0.5)
    glVertex2f( 1.0, -0.5)
    glVertex2f( 1.0,  0.5)
    glVertex2f(-1.0,  0.5)
    glEnd()
    
    glBegin(GL_LINES)
    glVertex2f(0.0, -0.5)
    glVertex2f(0.0,  0.5)
    glEnd()

    desenhar_metade_domino(valor_esq, -0.5, 0.0)
    desenhar_metade_domino(valor_dir, 0.5, 0.0)

def renderizar_cena(estado, renderizador_espacial):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    glTranslatef(0.0, 0.0, -15.0) 
    
    glColor3f(0.1, 0.5, 0.2)
    glBegin(GL_QUADS)
    glVertex2f(-20.0, -20.0)
    glVertex2f( 20.0, -20.0)
    glVertex2f( 20.0,  20.0)
    glVertex2f(-20.0,  20.0)
    glEnd()

    cadeia_visual = estado.get("cadeia_visual", [])
    layout_visual = renderizador_espacial.calcular_layout(cadeia_visual)

    for info in layout_visual:
        peca = info["peca"]
        pos_x = info["pos_x"]
        pos_y = info["pos_y"]
        angulo_base = info["angulo"]

        glPushMatrix()
        glTranslatef(pos_x, pos_y, 0.0)
        
        # Apply structural orientation first (horizontal vs vertical)
        glRotatef(angulo_base, 0.0, 0.0, 1.0)

        # Calculate visual flip accumulation
        rotacao_flip = 0
        
        if info.get("lado_jogado") == 1 and peca[1] == info.get("valor_conectado"):
             rotacao_flip += 180
        elif info.get("lado_jogado") == 0 and peca[0] == info.get("valor_conectado"):
             rotacao_flip += 180
        
        if info.get("dobrado"):
             rotacao_flip += 180
             
        # Resolve redundant flips (e.g., 360 degrees becomes 0)
        rotacao_flip = rotacao_flip % 360
        
        if rotacao_flip != 0:
             glRotatef(rotacao_flip, 0.0, 0.0, 1.0)

        desenhar_domino(peca[0], peca[1])
        glPopMatrix()


