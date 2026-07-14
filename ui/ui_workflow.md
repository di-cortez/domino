# UI Workflow

The visual interface has three layers:

1. input and game control: keyboard, pause, history, restart, human moves, AI turns;
2. board layout: tile positions, angles, and left/right branches;
3. drawing: OpenGL/Pygame calls for the board, tiles, HUD, menu, and text.

`DominoEngine` remains the source of game rules. The UI asks whether an action
is legal and then submits it; it does not reimplement domino legality.

## Main Entry Point

`visual_main.py`:

- initializes Pygame and the OpenGL camera;
- creates `DominoEngine`;
- builds selected agents through `create_agent_by_type`;
- creates `GameManager`, `GameController`, and `HudRenderer`;
- runs the 60 FPS loop: process input, update controller timers, render board,
  render HUD, flip display.

## Controller

`game_controller.py` owns UI state:

- visual snapshot history for stepping backward and forward;
- pause and speed;
- automatic AI turns through `GameManager`;
- human-turn stopping and keyboard routing;
- restart confirmation;
- temporary notifications for the HUD;
- menu state and agent type changes.

The controller uses two mixins:

- `human_control.py` for human tile/end selection and action execution;
- `hand_visibility.py` for visible/hidden hand rules.

## Agent Factory

`ui_agents.py` maps menu strings to real agent objects:

- `neural` -> `NeuralAgent`;
- `random_nn` -> `RandomNeuralAgent` with fixed, untrained weights;
- `heuristic` -> `StrategicAgent`;
- `random` -> `RandomUIAgent`;
- `human` -> `BlockedHumanAgent`;
- `rl` -> `RLAgent`.

`BlockedHumanAgent` is a sentinel. Human actions are executed directly by the
controller after keyboard input; `GameManager` should never call that object.

## Human Turn Flow

When the current player is human:

1. automatic advancement stops;
2. the controller selects the first playable tile when possible;
3. the HUD shows the hand and selection arrow;
4. Left/Right change the selected tile;
5. Up/Down/Tab switch the target end when both ends are legal;
6. Enter, D, or P submits play/draw/pass;
7. the engine validates and applies the action;
8. if the next player is AI, automatic advancement resumes.

The yellow arrow points to the tile half that connects to the board. It can be
`above` or `below`, independent of whether the selected end is left or right.

## Hand Visibility

Visibility rules:

- AI vs. AI: both hands are always visible;
- Human vs. AI: the human hand is always visible, and the AI hand can be
  toggled with J/K;
- Human vs. human: only the current player's hand is visible.

When a toggle is not allowed, the controller shows a short notification instead
of silently ignoring the key.

## Board Rendering

`scene_renderer.py` rebuilds a left-to-right tile chain from `board_history`,
asks `StateRenderer` for a stable pivot, and splits the chain into left and
right branches.

`layout_domino.py` calculates each branch without drawing:

- path segment direction;
- tile angle;
- pip order;
- inline position;
- corner position when a branch reaches a board limit.

`domino_drawing.py` applies the OpenGL transform for one tile and delegates the
primitive drawing to `primitives.py`.

## Tests

`test_ui_controller.py` is executable without pytest:

```bash
python ui/test_ui_controller.py
```

The tests cover automatic advancement, human-turn blocking, selection
navigation, end switching, arrow position, human play/draw/pass rejection,
restart behavior, menu agent cycling, draw notifications, and hand visibility.
