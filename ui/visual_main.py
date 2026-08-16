"""
Entry point for the visual interface.

This file creates the window, engine, agents, controller, and render loop. Game
rules stay in the middleware/engine; UI interaction stays in `GameController`.
"""

import argparse

import pygame
from OpenGL.GL import GL_MODELVIEW, GL_PROJECTION, glMatrixMode
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, OPENGL

from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from ui.game_controller import GameController
from ui.hud import HudRenderer
from ui.scene_renderer import render_scene
from ui.ui_agents import agent_type_name, create_agent_by_type
from middleware.rulesets import DEFAULT_RULESET_NAME, RULESET_NAMES


def _window_caption(agent_types, ruleset=DEFAULT_RULESET_NAME):
    """Return a caption that reflects the currently selected matchup."""
    names = [agent_type_name(agent_type) for agent_type in agent_types]
    prefix = "Domino" if ruleset == DEFAULT_RULESET_NAME else f"Domino [{ruleset}]"
    return f"{prefix} - {names[0]} vs {names[1]}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the visual domino simulator.")
    parser.add_argument(
        "--ruleset",
        choices=RULESET_NAMES,
        default=DEFAULT_RULESET_NAME,
    )
    args = parser.parse_args(argv)
    pygame.init()
    display = (1024, 768)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    engine = DominoEngine(player_count=2, ruleset=args.ruleset)

    agent_types = ["neural", "heuristic"]
    agents = [
        create_agent_by_type(agent_type, args.ruleset)
        for agent_type in agent_types
    ]

    manager = GameManager(engine, agents)
    hud = HudRenderer(args.ruleset)

    controller = GameController(
        manager,
        engine,
        interval_ms=1000,
        agent_types=agent_types,
    )

    last_caption = _window_caption(controller.agent_types, args.ruleset)
    pygame.display.set_caption(last_caption)

    print(
        f"P0: {agent_type_name(agent_types[0])} | "
        f"P1: {agent_type_name(agent_types[1])}"
    )
    print("M: menu | Space: pause | Left/Right: history step | ESC: quit")

    clock = pygame.time.Clock()
    while True:
        dt_ms = clock.tick(60)

        if not controller.process_input():
            pygame.quit()
            return

        controller.update(dt_ms)

        caption = _window_caption(controller.agent_types, args.ruleset)
        if caption != last_caption:
            pygame.display.set_caption(caption)
            last_caption = caption

        render_scene(controller.current_state())
        hud.render(controller.current_state(), controller, display)
        pygame.display.flip()


if __name__ == "__main__":
    main()
