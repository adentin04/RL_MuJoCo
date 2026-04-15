"""
Replay buffer for TD-MPC2 — stores full episode trajectories
and samples random trajectory chunks for training.
"""

import numpy as np


class ReplayBuffer:
    """Stores transitions and samples random chunks of consecutive steps."""

    def __init__(self, capacity, state_dim, action_dim, goal_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, goal, done):
        i = self.idx % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.goals[i] = goal
        self.dones[i] = done
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample_chunks(self, batch_size, seq_len):
        """
        Sample B chunks of length T from the buffer.

        Returns:
            states:  [B, T+1, state_dim]  (T transitions need T+1 states)
            actions: [B, T, action_dim]
            rewards: [B, T]
            goals:   [B, goal_dim]
        """
        # Find valid start indices (not crossing episode boundaries or buffer end)
        max_start = self.size - seq_len - 1
        if max_start < 1:
            raise ValueError(f"Not enough data: {self.size} < {seq_len + 2}")

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_goals = []

        attempts = 0
        while len(batch_states) < batch_size and attempts < batch_size * 10:
            start = np.random.randint(0, max_start)
            attempts += 1

            # Skip if chunk crosses an episode boundary
            chunk_dones = self.dones[start: start + seq_len]
            if np.any(chunk_dones):
                continue

            batch_states.append(self.states[start: start + seq_len + 1])
            batch_actions.append(self.actions[start: start + seq_len])
            batch_rewards.append(self.rewards[start: start + seq_len])
            batch_goals.append(self.goals[start])

        B = len(batch_states)
        if B == 0:
            raise ValueError("Could not sample valid chunks (too many episode boundaries)")

        return (
            np.stack(batch_states),   # [B, T+1, state_dim]
            np.stack(batch_actions),  # [B, T, action_dim]
            np.stack(batch_rewards),  # [B, T]
            np.stack(batch_goals),    # [B, goal_dim]
        )
