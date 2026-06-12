from OpenGL.GL import *
from OpenGL.GLU import *
import math
import pygame

class HudRenderer:
    """Renders the 2D HUD overlay (turn indicator, notifications, config menu)."""

    _NOTIF_DURACAO_MS = 3000
    _NOTIF_FADE_MS    = 1000
    _MENU_PADDING     = 44
    _MENU_ITEM_H      = 50
    _MENU_HEADER_H    = 68
    _MENU_FOOTER_H    = 30

    def __init__(self):
        self._fontes_prontas = False
        self._f_titulo = None
        self._f_normal = None
        self._f_dica   = None

    # ------------------------------------------------------------------ #
    # Font init (deferred until display is up)
    # ------------------------------------------------------------------ #

    def _init_fontes(self):
        if self._fontes_prontas:
            return
        pygame.font.init()
        self._f_titulo = pygame.font.SysFont("monospace", 26, bold=True)
        self._f_normal = pygame.font.SysFont("monospace", 20)
        self._f_dica   = pygame.font.SysFont("monospace", 14)
        self._fontes_prontas = True

    # ------------------------------------------------------------------ #
    # Low-level OpenGL helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _upload_texture(surface):
        data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return tid, w, h

    def _rect(self, x, y, w, h, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_QUADS)
        glVertex2f(x,     y);     glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x,     y + h)
        glEnd()

    def _texto(self, texto, fonte, cor, x, y, alpha=1.0):
        """Draw text at screen-space (x, y) top-left. Returns (w, h)."""
        surf = fonte.render(texto, True, cor).convert_alpha()
        tid, w, h = self._upload_texture(surf)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, alpha)
        glBindTexture(GL_TEXTURE_2D, tid)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x,     y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x,     y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures(1, [tid])
        return w, h

    def _begin_2d(self, sw, sh):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, sw, sh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _end_2d(self):
        glDisable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ------------------------------------------------------------------ #
    # Public render entry-point
    # ------------------------------------------------------------------ #

    def renderizar(self, estado, controlador, display):
        self._init_fontes()
        sw, sh = display
        self._begin_2d(sw, sh)

        jogador_atual = estado.get("jogador_atual", 0)
        turno         = estado.get("turno", 0)
        tamanhos      = estado.get("tamanhos_maos", [7, 7])
        nomes = ["Neural" if t == "neural" else "Heuristico"
                 for t in controlador.tipos_agentes]

        # --- top bar ---
        bar_h = 38
        self._rect(0, 0, sw, bar_h, 0, 0, 0, 0.75)

        # J0 (left) — name + tile count + [VEZ] badge
        cor0 = (80, 200, 255) if jogador_atual == 0 else (110, 110, 110)
        lbl0 = f"J0 ({nomes[0]}) [{tamanhos[0]}]"
        if jogador_atual == 0:
            lbl0 += "  [VEZ]"
            lw0, _ = self._f_normal.size(lbl0)
            self._rect(5, 4, lw0 + 10, 30, 0.1, 0.4, 0.7, 0.8)
        self._texto(lbl0, self._f_normal, cor0, 12, 9)

        # turn counter (centre)
        turno_txt = f"Turno {turno}"
        tw, _ = self._f_normal.size(turno_txt)
        self._texto(turno_txt, self._f_normal, (220, 220, 170), sw // 2 - tw // 2, 9)

        # J1 (right) — name + tile count + [VEZ] badge
        cor1 = (255, 185, 55) if jogador_atual == 1 else (110, 110, 110)
        lbl1 = f"J1 ({nomes[1]}) [{tamanhos[1]}]"
        if jogador_atual == 1:
            lbl1 = "[VEZ]  " + lbl1
        lw1, _ = self._f_normal.size(lbl1)
        if jogador_atual == 1:
            self._rect(sw - lw1 - 15, 4, lw1 + 10, 30, 0.5, 0.28, 0.05, 0.8)
        self._texto(lbl1, self._f_normal, cor1, sw - lw1 - 10, 9)

        # --- bottom hint bar ---
        dica = "M: Menu | Espaco: Pausa | < >: Passo | ESC: Sair"
        dw, dh = self._f_dica.size(dica)
        self._rect(0, sh - dh - 6, sw, dh + 6, 0, 0, 0, 0.6)
        self._texto(dica, self._f_dica, (150, 150, 150), sw // 2 - dw // 2, sh - dh - 3)

        # --- draw notification ---
        if controlador.notificacao:
            self._renderizar_notificacao(controlador.notificacao, display)

        # --- game-over banner ---
        if controlador.fim_de_jogo and controlador.info_final:
            v = controlador.info_final.get("vencedor")
            if v == -1:
                msg = "EMPATE!"
            elif v is not None:
                msg = f"Fim! Vencedor: J{v} ({nomes[v]})"
            else:
                msg = "Fim de jogo"
            mw, mh = self._f_titulo.size(msg)
            bx = sw // 2 - mw // 2 - 14
            by = sh // 2 - mh // 2 - 10
            self._rect(bx, by, mw + 28, mh + 20, 0, 0, 0, 0.88)
            self._rect(bx, by, mw + 28, 3, 0.85, 0.6, 0.1, 1.0)
            self._texto(msg, self._f_titulo, (255, 220, 50),
                        sw // 2 - mw // 2, sh // 2 - mh // 2)
            sub = "Pressione M > Reiniciar para nova partida"
            sw2, _ = self._f_dica.size(sub)
            self._texto(sub, self._f_dica, (170, 170, 170),
                        sw // 2 - sw2 // 2, sh // 2 + mh // 2 + 8)

        # --- config menu ---
        if controlador.menu_aberto:
            self._renderizar_menu(controlador, display, nomes)

        self._end_2d()

    # ------------------------------------------------------------------ #
    # Draw-from-stock notification
    # ------------------------------------------------------------------ #

    def _renderizar_notificacao(self, notif, display):
        sw, _sh = display
        texto = notif["texto"]
        tempo = notif["tempo_ms"]
        alpha = min(1.0, tempo / self._NOTIF_FADE_MS)

        tw, th = self._f_normal.size(texto)
        pad_x, pad_y = 16, 7
        bx = sw // 2 - tw // 2 - pad_x
        by = 46
        bw = tw + pad_x * 2
        bh = th + pad_y * 2

        self._rect(bx, by, bw, bh, 0.04, 0.04, 0.04, 0.85 * alpha)
        self._rect(bx, by, bw, 2, 0.9, 0.55, 0.1, alpha)
        self._texto(texto, self._f_normal, (255, 200, 80), bx + pad_x, by + pad_y, alpha)

    # ------------------------------------------------------------------ #
    # Config menu (dimensions calculated from content)
    # ------------------------------------------------------------------ #

    def _renderizar_menu(self, controlador, display, nomes):
        sw, sh = display

        def lbl_tipo(tipo):
            return f"[ {tipo.capitalize()} ]"

        itens = [
            f"Jogador 0:  {lbl_tipo(controlador.tipos_agentes[0])}  (Enter: alternar)",
            f"Jogador 1:  {lbl_tipo(controlador.tipos_agentes[1])}  (Enter: alternar)",
            "Reiniciar Partida",
        ]
        footer = "Setas: navegar  |  Enter/Espaco: selecionar  |  M / ESC: fechar"

        item_ws  = [self._f_normal.size("> " + t)[0] for t in itens]
        title_w  = self._f_titulo.size("CONFIGURACAO")[0]
        footer_w = self._f_dica.size(footer)[0]
        mw = max(*item_ws, title_w, footer_w) + self._MENU_PADDING
        mh = self._MENU_HEADER_H + len(itens) * self._MENU_ITEM_H + self._MENU_FOOTER_H

        mx = (sw - mw) // 2
        my = (sh - mh) // 2

        self._rect(0, 0, sw, sh, 0, 0, 0, 0.55)
        self._rect(mx, my, mw, mh, 0.07, 0.07, 0.12, 0.97)
        self._rect(mx, my,          mw, 3, 0.25, 0.5, 1.0, 1.0)
        self._rect(mx, my + mh - 3, mw, 3, 0.25, 0.5, 1.0, 1.0)

        self._texto("CONFIGURACAO", self._f_titulo, (180, 210, 255), mx + 16, my + 14)

        for i, item in enumerate(itens):
            iy  = my + self._MENU_HEADER_H + i * self._MENU_ITEM_H
            sel = (i == controlador.menu_cursor)
            if sel:
                self._rect(mx + 8, iy - 5, mw - 16, 34, 0.15, 0.35, 0.75, 0.85)
            cor    = (255, 255, 110) if sel else (200, 200, 210)
            prefix = "> " if sel else "  "
            self._texto(prefix + item, self._f_normal, cor, mx + 16, iy)

        fw, _ = self._f_dica.size(footer)
        self._texto(footer, self._f_dica, (120, 120, 145),
                    mx + mw // 2 - fw // 2, my + mh - 22)
