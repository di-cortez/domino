"""Uniform-random baseline agent."""

import random

from middleware.middleware import Agent


class RandomAgent(Agent):
    """Choose uniformly from the legal action list."""

    def choose_move(self, state, legal_actions):
        return random.choice(legal_actions)
