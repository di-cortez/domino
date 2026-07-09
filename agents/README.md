# Agents

All playable agents expose the same `choose_move(state, legal_actions)` shape so
`GameManager` can run any pair without knowing how each decision is made.

| File | Purpose |
|---|---|
| `agent.py` | Baseline `RandomAgent` and `GreedyAgent`, plus a small runnable demo. |
| `encoder.py` | Single source of truth for state-to-vector and action-to-index encoding. |
| `heuristic_agent.py` | `StrategicAgent`, the rule-based teacher used for supervised labels and benchmarks. |
| `nn.py` | Supervised MLP backend. Uses CuPy automatically when installed, otherwise NumPy. |
| `neural_agent.py` | Loads `models/domino_sl_weights.npz` and plays the supervised policy. |
| `rl_nn.py` | Policy network with REINFORCE gradients, entropy regularization, and value baseline. |
| `rl_agent.py` | Wraps `PolicyNetwork` for training trajectories or greedy evaluation play. |

## State And Action Encoding

`DominoEncoder` produces an 86-dimensional input vector:

- current player hand, 28 bits;
- board ends, 14 bits;
- stock size, 1 value;
- both hand sizes, 2 values;
- normalized turn count, 1 value;
- opponent dead suits inferred from draw/pass history, 7 bits;
- additional board/action context features used by the neural agents.

The output space has 58 actions:

- 28 tile actions on the left end;
- 28 tile actions on the right end;
- one draw action;
- one pass action.

All neural and RL code uses this encoder, which keeps training, inference, and
diagnostics aligned.
