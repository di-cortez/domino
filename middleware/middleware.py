"""Glue layer between the rules engine and interchangeable agents."""


class Agent:
    """Minimal protocol implemented by every automatic player."""

    def choose_move(self, state, legal_actions):
        """Return one action from ``legal_actions`` for the current state."""
        raise NotImplementedError("Agents must implement choose_move().")


class GameManager:
    """Advance a game by asking the current agent for one legal action."""

    def __init__(self, engine, agents):
        if len(agents) != engine.player_count:
            raise ValueError("The number of agents must match the number of players.")
        self.engine = engine
        self.agents = agents
        self.training_history = []

    def play_turn(self):
        state = self.engine._get_state()
        current_player = state["current_player"]
        legal_actions = self.engine.valid_actions(current_player)
        chosen_action = self.agents[current_player].choose_move(state, legal_actions)

        self.training_history.append({
            "state": state,
            "target_action": chosen_action,
        })

        _, game_over, info = self.engine.step(chosen_action)
        info["action"] = chosen_action
        info["acting_player"] = current_player
        return game_over, info

    def play_full_game(self):
        self.training_history = []
        game_over = False
        info = {}

        while not game_over:
            game_over, info = self.play_turn()

        return info, self.training_history
