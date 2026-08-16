"""Immutable records for same-iteration opponent-decision restarts."""

from dataclasses import dataclass

from middleware.domino_engine import DominoRestartState


@dataclass(frozen=True)
class CapturedOpponentDecision:
    """One eligible pre-action state captured inside a normal rollout."""

    snapshot_ordinal: int
    source_turn: int
    original_learner_position: int
    source_legal_tile_action_count: int
    engine_state: DominoRestartState

    @property
    def restart_learner_position(self):
        return 1 - self.original_learner_position

@dataclass(frozen=True)
class OpponentDecisionRestart:
    """Captured state enriched with its source assignment and stable identity."""

    restart_index: int
    source_iteration: int
    source_game_index: int
    snapshot_ordinal: int
    source_turn: int
    original_learner_position: int
    source_legal_tile_action_count: int
    opponent_kind: str
    opponent_id: str | None
    bucket_name: str
    bank_slot: int | None
    engine_state: DominoRestartState

    @property
    def restart_learner_position(self):
        return 1 - self.original_learner_position
