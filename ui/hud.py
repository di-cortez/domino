"""
HUD 2D desenhada por cima da mesa OpenGL.

Este módulo só lê o estado do controlador. Ele não executa jogadas nem altera
o motor; toda mudança de partida passa por `ControladorPartida`.
"""

import pygame

from ui.primitivas import (
    desenhar_domino_2d,
    desenhar_texto,
    finalizar_2d,
    iniciar_2d,
    retangulo,
    triangulo,
)
from ui.agentes_ui import nome_tipo_agente


class HudRenderer:
    """Renderiza turno, mãos, monte, notificações, fim de jogo e menu."""

    _NOTIF_DURACAO_MS = 3000
    _NOTIF_FADE_MS    = 1000

    _BAR_H            = 38
    _MAOS_H           = 74

    _DOMINO_W         = 22
    _DOMINO_H         = 42
    _DOMINO_GAP       = 4

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
    # Fontes
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
    # Entrada pública da HUD
    # ------------------------------------------------------------------ #

    def renderizar(self, estado, controlador, display):
        self._init_fontes()

        sw, sh = display

        iniciar_2d(sw, sh)

        jogador_atual = estado.get("jogador_atual", 0)
        turno = estado.get("turno", 0)

        nomes = [
            nome_tipo_agente(tipo)
            for tipo in controlador.tipos_agentes
        ]

        self._renderizar_barra_superior(jogador_atual, turno, nomes, display)
        self._renderizar_barra_maos(estado, controlador, display)
        self._renderizar_barra_inferior(display, controlador.jogador_humano_ativo())

        if controlador.notificacao:
            self._renderizar_notificacao(controlador.notificacao, display)

        if controlador.fim_de_jogo and controlador.info_final:
            self._renderizar_fim_de_jogo(controlador, nomes, display)

        if controlador.menu_aberto:
            self._renderizar_menu(controlador, display)

        finalizar_2d()

    # ------------------------------------------------------------------ #
    # Barra superior
    # ------------------------------------------------------------------ #

    def _renderizar_barra_superior(self, jogador_atual, turno, nomes, display):
        sw, _sh = display

        retangulo(0, 0, sw, self._BAR_H, (0, 0, 0), 0.78)

        # Jogador 0
        cor0 = (80, 200, 255) if jogador_atual == 0 else (110, 110, 110)
        lbl0 = f"J0 ({nomes[0]})"

        if jogador_atual == 0:
            lbl0 += "  [VEZ]"
            lw0, _ = self._f_normal.size(lbl0)
            retangulo(5, 4, lw0 + 10, 30, (0.1, 0.4, 0.7), 0.8)

        desenhar_texto(lbl0, self._f_normal, cor0, 12, 9)

        # Turno
        turno_txt = f"Turno {turno}"
        tw, _ = self._f_normal.size(turno_txt)

        desenhar_texto(
            turno_txt,
            self._f_normal,
            (220, 220, 170),
            sw // 2 - tw // 2,
            9,
        )

        # Jogador 1
        cor1 = (255, 185, 55) if jogador_atual == 1 else (110, 110, 110)
        lbl1 = f"J1 ({nomes[1]})"

        if jogador_atual == 1:
            lbl1 = "[VEZ]  " + lbl1

        lw1, _ = self._f_normal.size(lbl1)

        if jogador_atual == 1:
            retangulo(sw - lw1 - 15, 4, lw1 + 10, 30, (0.5, 0.28, 0.05), 0.8)

        desenhar_texto(lbl1, self._f_normal, cor1, sw - lw1 - 10, 9)

    # ------------------------------------------------------------------ #
    # Barra inferior
    # ------------------------------------------------------------------ #

    def _renderizar_barra_inferior(self, display, humano_ativo):
        sw, sh = display

        if humano_ativo:
            dica = (
                "Esq/Dir: peça | Cima/Baixo: ponta | Enter: jogar | "
                "C: comprar | P: passar | M: menu | ESC: sair"
            )
        else:
            dica = "M: Menu | R:Reiniciar | Espaco: Pausa | Setas:Passo | +-:velocidade | ESC: Sair"

        dw, dh = self._f_dica.size(dica)

        retangulo(0, sh - dh - 6, sw, dh + 6, (0, 0, 0), 0.6)

        desenhar_texto(
            dica,
            self._f_dica,
            (150, 150, 150),
            sw // 2 - dw // 2,
            sh - dh - 3,
        )

    # ------------------------------------------------------------------ #
    # Mãos e monte
    # ------------------------------------------------------------------ #

    def _desenhar_seta_selecao(self, x, y, w, h, posicao):
        cx = x + w / 2.0

        if posicao == "cima":
            ponta_y = y - 4
            base_y = y - 14

            triangulo(
                cx,
                ponta_y,
                cx - 7,
                base_y,
                cx + 7,
                base_y,
                (1.0, 0.82, 0.08),
                1.0,
            )
            return

        topo = y + h + 4
        base = topo + 10

        triangulo(
            cx,
            topo,
            cx - 7,
            base,
            cx + 7,
            base,
            (1.0, 0.82, 0.08),
            1.0,
        )

    def _renderizar_mao(
        self,
        pecas,
        x,
        y,
        max_w,
        max_h,
        align="left",
        indice_selecionado=None,
        posicao_seta=None,
        oculta=False,
    ):
        if not pecas:
            return

        # A mão é desenhada em miniatura e com limite de espaço. Se houver
        # mais peças do que cabem, mostramos as primeiras e um contador "+N".
        count = len(pecas)

        w = self._DOMINO_W
        h = self._DOMINO_H
        gap = self._DOMINO_GAP

        if h > max_h:
            escala = max(0.72, max_h / h)
            w = int(self._DOMINO_W * escala)
            h = int(self._DOMINO_H * escala)
            gap = max(3, int(self._DOMINO_GAP * escala))

        cols = max(1, int((max_w + gap) // (w + gap)))
        capacidade = min(count, cols)

        visiveis = pecas[:capacidade]
        ocultas = count - len(visiveis)

        total_w = len(visiveis) * w + max(0, len(visiveis) - 1) * gap

        if align == "left":
            start_x = x
        else:
            start_x = x + max_w - total_w

        for idx, peca in enumerate(visiveis):
            px = start_x + idx * (w + gap)
            py = y

            if oculta:
                desenhar_domino_2d(px, py, w, h, verso=True)
            else:
                desenhar_domino_2d(px, py, w, h, peca=tuple(peca))

            # A seta só aparece para mão visível; se a mão está oculta, a HUD
            # não deve revelar qual peça o humano selecionou.
            if not oculta and indice_selecionado == idx:
                self._desenhar_seta_selecao(px, py, w, h, posicao_seta)

        if ocultas > 0:
            txt = f"+{ocultas}"
            tw, th = self._f_dica.size(txt)

            tx = start_x + total_w - tw
            ty = y + max_h - th

            desenhar_texto(txt, self._f_dica, (230, 230, 230), tx, ty)

    def _renderizar_barra_maos(self, estado, controlador, display):
        sw, _sh = display

        y = self._BAR_H

        retangulo(0, y, sw, self._MAOS_H, (0.02, 0.08, 0.05), 0.82)
        retangulo(0, y + self._MAOS_H - 1, sw, 1, (0.05, 0.20, 0.12), 0.9)

        maos = estado.get("maos") or []

        mao0 = maos[0] if len(maos) > 0 else []
        mao1 = maos[1] if len(maos) > 1 else []

        margem = 12
        centro_w = 184

        area_w = max(210, (sw - centro_w - margem * 4) // 2)

        mao_y = y + 13
        mao_h = self._MAOS_H - 18

        jogador_atual = estado.get("jogador_atual", 0)
        humano_ativo = controlador.jogador_humano_ativo()

        sel0 = None
        sel1 = None

        if humano_ativo and jogador_atual == 0:
            sel0 = controlador.indice_peca_selecionada
        elif humano_ativo and jogador_atual == 1:
            sel1 = controlador.indice_peca_selecionada

        # O controlador informa se a seta fica acima ou abaixo da peça. A HUD
        # só cuida do desenho, não da regra de encaixe.
        posicao_seta = controlador.posicao_seta_peca_selecionada()
        oculta0 = controlador.mao_oculta(0)
        oculta1 = controlador.mao_oculta(1)

        self._renderizar_mao(
            mao0,
            margem,
            mao_y,
            area_w,
            mao_h,
            align="left",
            indice_selecionado=sel0,
            posicao_seta=posicao_seta,
            oculta=oculta0,
        )
        self._renderizar_mao(
            mao1,
            sw - margem - area_w,
            mao_y,
            area_w,
            mao_h,
            align="right",
            indice_selecionado=sel1,
            posicao_seta=posicao_seta,
            oculta=oculta1,
        )

        self._renderizar_monte(estado, display)

    def _renderizar_monte(self, estado, display):
        sw, _sh = display

        y = self._BAR_H

        monte_n = estado.get("monte_tamanho", len(estado.get("monte", [])))

        label = "Monte"
        count_txt = f"x {monte_n}"

        tile_w = 40
        tile_h = 22
        gap = 10

        label_w, label_h = self._f_dica.size(label)
        count_w, count_h = self._f_normal.size(count_txt)

        total_w = label_w + gap + tile_w + gap + count_w

        start_x = sw // 2 - total_w // 2

        tile_x = start_x + label_w + gap
        tile_y = y + self._MAOS_H // 2 - tile_h // 2 + 2

        text_y = tile_y + tile_h // 2 - label_h // 2
        count_y = tile_y + tile_h // 2 - count_h // 2

        desenhar_texto(
            label,
            self._f_dica,
            (190, 215, 190),
            start_x,
            text_y,
        )

        desenhar_domino_2d(tile_x, tile_y, tile_w, tile_h, verso=True)

        desenhar_texto(
            count_txt,
            self._f_normal,
            (220, 230, 210),
            tile_x + tile_w + gap,
            count_y,
        )

    # ------------------------------------------------------------------ #
    # Notificação
    # ------------------------------------------------------------------ #

    def _renderizar_notificacao(self, notif, display):
        sw, _sh = display

        texto = notif["texto"]
        tempo = notif["tempo_ms"]

        alpha = min(1.0, tempo / self._NOTIF_FADE_MS)

        tw, th = self._f_normal.size(texto)

        pad_x = 16
        pad_y = 7

        bx = sw // 2 - tw // 2 - pad_x
        by = self._BAR_H + self._MAOS_H + 8

        bw = tw + pad_x * 2
        bh = th + pad_y * 2

        retangulo(bx, by, bw, bh, (0.04, 0.04, 0.04), 0.85 * alpha)
        retangulo(bx, by, bw, 2, (0.9, 0.55, 0.1), alpha)

        desenhar_texto(
            texto,
            self._f_normal,
            (255, 200, 80),
            bx + pad_x,
            by + pad_y,
            alpha,
        )

    # ------------------------------------------------------------------ #
    # Fim de jogo
    # ------------------------------------------------------------------ #

    def _renderizar_fim_de_jogo(self, controlador, nomes, display):
        sw, sh = display

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

        retangulo(bx, by, mw + 28, mh + 20, (0, 0, 0), 0.88)
        retangulo(bx, by, mw + 28, 3, (0.85, 0.6, 0.1), 1.0)

        desenhar_texto(
            msg,
            self._f_titulo,
            (255, 220, 50),
            sw // 2 - mw // 2,
            sh // 2 - mh // 2,
        )

        sub = "Pressione M > Reiniciar para nova partida"
        sw2, _ = self._f_dica.size(sub)

        desenhar_texto(
            sub,
            self._f_dica,
            (170, 170, 170),
            sw // 2 - sw2 // 2,
            sh // 2 + mh // 2 + 8,
        )

    # ------------------------------------------------------------------ #
    # Menu
    # ------------------------------------------------------------------ #

    def _renderizar_menu(self, controlador, display):
        sw, sh = display

        def lbl_tipo(tipo):
            return f"[ {tipo.capitalize()} ]"

        itens = [
            f"Jogador 0:  {lbl_tipo(controlador.tipos_agentes[0])}  (Enter: alternar)",
            f"Jogador 1:  {lbl_tipo(controlador.tipos_agentes[1])}  (Enter: alternar)",
            "Reiniciar Partida",
        ]

        footer = "Setas: navegar  |  Enter/Espaco: selecionar  |  M / ESC: fechar"

        item_ws = [
            self._f_normal.size("> " + item)[0]
            for item in itens
        ]

        title_w = self._f_titulo.size("CONFIGURACAO")[0]
        footer_w = self._f_dica.size(footer)[0]

        mw = max(*item_ws, title_w, footer_w) + self._MENU_PADDING
        mh = self._MENU_HEADER_H + len(itens) * self._MENU_ITEM_H + self._MENU_FOOTER_H

        mx = (sw - mw) // 2
        my = (sh - mh) // 2

        retangulo(0, 0, sw, sh, (0, 0, 0), 0.55)

        retangulo(mx, my, mw, mh, (0.07, 0.07, 0.12), 0.97)
        retangulo(mx, my, mw, 3, (0.25, 0.5, 1.0), 1.0)
        retangulo(mx, my + mh - 3, mw, 3, (0.25, 0.5, 1.0), 1.0)

        desenhar_texto(
            "CONFIGURACAO",
            self._f_titulo,
            (180, 210, 255),
            mx + 16,
            my + 14,
        )

        for i, item in enumerate(itens):
            iy = my + self._MENU_HEADER_H + i * self._MENU_ITEM_H

            selecionado = i == controlador.menu_cursor

            if selecionado:
                retangulo(mx + 8, iy - 5, mw - 16, 34, (0.15, 0.35, 0.75), 0.85)

            cor = (255, 255, 110) if selecionado else (200, 200, 210)
            prefixo = "> " if selecionado else "  "

            desenhar_texto(
                prefixo + item,
                self._f_normal,
                cor,
                mx + 16,
                iy,
            )

        fw, _ = self._f_dica.size(footer)

        desenhar_texto(
            footer,
            self._f_dica,
            (120, 120, 145),
            mx + mw // 2 - fw // 2,
            my + mh - 22,
        )
