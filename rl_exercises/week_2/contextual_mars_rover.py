from __future__ import annotations

import numpy as np
from rl_exercises.environments import MarsRover


class ContextualMarsRover(MarsRover):
    """
    Contextual version of the MarsRover environment.
    The context modifies the transition dynamics (via slip probability)
    and the reward structure (via the goal reward value).
    """

    def __init__(self, context: dict | None = None, **kwargs):
        self.context = (
            context if context is not None else {"slip": 0.0, "goal_reward": 10.0}
        )

        # Calculate success probability from context
        p_success = 1.0 - self.context.get("slip", 0.0)
        n_states = 5
        transition_probs = np.full((n_states, 2), p_success)

        # Define rewards based on context
        rewards = [1.0, 0.0, 0.0, 0.0, float(self.context.get("goal_reward", 10.0))]

        super().__init__(
            transition_probabilities=transition_probs, rewards=rewards, **kwargs
        )

    def set_context(self, context: dict):
        """
        Updates the internal state to reflect a new context.
        Recomputes the transition matrix and reward vector.
        """
        self.context = context
        p_success = 1.0 - context.get("slip", 0.0)

        # Update internal P and rewards list from base class
        self.P = np.full((self.observation_space.n, 2), p_success)
        self.rewards = [1.0, 0.0, 0.0, 0.0, float(context.get("goal_reward", 10.0))]

        # Regenerate the transition matrix T[s, a, s']
        self.transition_matrix = self.get_transition_matrix()
