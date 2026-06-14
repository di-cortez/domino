"""
Ponto de entrada da interface visual.

Este arquivo monta a janela, cria motor/agentes/controlador e mantém o laço de
renderização. A regra de jogo fica no middleware/motor; a regra de interação
fica em `ControladorPartida`.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Módulos da arquitetura
from middleware.motor_domino import MotorDomino
from middleware.middleware import GerenciadorPartida
from ui.agentes_ui import criar_agente_por_tipo
from ui.interface import renderizar_cena
from ui.hud import HudRenderer
from ui.controle_partida import ControladorPartida


def main():
    # 1. Inicialização do Pygame e da janela OpenGL
    pygame.init()
    display = (1024, 768)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Dominó — Neural vs Heurístico")

    # Configuração da câmera OpenGL
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    # 2. Backend (motor, agentes, renderizador).
    # A fábrica de agentes fica em `agentes_ui.py`, para o menu e o main
    # usarem os mesmos nomes/tipos.
    motor = MotorDomino(num_jogadores=2)

    # print(motor._obter_estado())

    tipos_agentes = ['neural', 'heuristico']
    agentes = [
        criar_agente_por_tipo(tipo)
        for tipo in tipos_agentes
    ]

    gerenciador = GerenciadorPartida(motor, agentes)
    hud = HudRenderer()

    # 3. Controle: teclado, avanço, retrocesso, pausa, menu e humanos.
    controlador = ControladorPartida(gerenciador, motor,
                                     intervalo_ms=1000,
                                     tipos_agentes=tipos_agentes)

    print("J0: Neural | J1: Heuristico")
    print("M: menu | Espaco: pausa | </>: passo | ESC: sair")

    # 4. Laço principal (não bloqueante: sem time.sleep)
    clock = pygame.time.Clock()
    while True:
        dt_ms = clock.tick(60)   # ms desde o último frame; mantém ~60 FPS

        if not controlador.processar_entrada():
            pygame.quit()
            return

        controlador.atualizar(dt_ms)

        # Renderiza o estado escolhido pelo controlador (pode ser do passado).
        renderizar_cena(controlador.estado_atual())
        hud.renderizar(controlador.estado_atual(), controlador, display)
        pygame.display.flip()


if __name__ == "__main__":
    main()
