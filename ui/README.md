# UI

The UI is a Pygame/OpenGL visual layer over `DominoEngine`. It never decides
whether a move is legal; it queries the engine and only manages interaction,
history, rendering, and HUD state.

For a step-by-step explanation of the UI flow, see `ui_workflow.md`.

| File | Purpose |
|---|---|
| `visual_main.py` | Main entry point: creates the window, engine, agents, controller, HUD, and render loop. |
| `game_controller.py` | UI orchestration: history, pause/speed, automatic turns, menu, restart confirmation, notifications. |
| `human_control.py` | Human-turn keyboard interaction: tile selection, end selection, play, draw, pass, and selection-arrow position. |
| `hand_visibility.py` | Rules for visible/hidden hands in AI-vs-AI, human-vs-AI, and human-vs-human modes. |
| `ui_agents.py` | Agent factory and display names for menu selections. |
| `hud.py` | 2D overlay with top bar, hands, stock, notifications, game-over banner, menu, and shortcut hints. |
| `scene_renderer.py` | 3D board renderer using a stable pivot and left/right branches. |
| `layout_domino.py` | Pure geometry: branch direction, tile angle, pip order, inline positions, and corner turns. |
| `domino_drawing.py` | OpenGL transform and drawing call for one 3D domino tile. |
| `primitives.py` | Small OpenGL/Pygame primitives: rectangles, lines, circles, text, and domino drawing. |
| `state_renderer.py` | Keeps the board pivot stable across snapshots. |
| `visual_config.py` | Board geometry constants. |
| `test_ui_controller.py` | Sequential UI/controller unit tests. |

Run:

```bash
python -m ui.visual_main
```

Run UI tests:

```bash
python ui/test_ui_controller.py
```
