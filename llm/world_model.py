"""
TD-MPC2 World Model — learns the physics of the robot in latent space.

Components:
    Encoder  h(s)        → z          maps state to latent
    Dynamics d(z, a, g)  → z'         predicts next latent (goal-conditioned)
    Reward   R(z, a, g)  → r̂          predicts reward
    Value    V(z, g)     → v̂          estimates state value (for planning)
    Policy   π(z, g)     → a          learned action prior (warm-starts MPPI)

Training losses:
    L_dyn   = ||d(z_t, a_t, g) − sg(h(s_{t+1}))||²    (dynamics consistency)
    L_rew   = ||R(z_t, a_t, g) − r_t||²               (reward prediction)
    L_val   = ||V(z_t, g) − R_t||²                     (value / TD target)
    L_pi    = -V(d(z_t, π(z_t, g), g), g)              (policy via value gradient)
"""

import torch
import torch.nn as nn


def _mlp(in_dim, hidden_dim, out_dim, layers=2):
    """Build a simple MLP with LayerNorm + Mish activations (TD-MPC2 style)."""
    mods = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
    mods.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    """TD-MPC2 world model, conditioned on a goal vector from the VLA encoder."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        z = cfg.latent_dim
        h = cfg.hidden_dim
        a = cfg.action_dim
        g = cfg.goal_dim

        # Encoder: state → latent
        self.encoder = _mlp(cfg.state_dim, h, z)

        # Dynamics: (z, a, g) → z'
        self.dynamics = _mlp(z + a + g, h, z)

        # Reward predictor: (z, a, g) → scalar
        self.reward = _mlp(z + a + g, h, 1)

        # Value function: (z, g) → scalar
        self.value = _mlp(z + g, h, 1)
        # Target value (EMA copy, no gradients)
        self.value_target = _mlp(z + g, h, 1)
        self.value_target.load_state_dict(self.value.state_dict())
        for p in self.value_target.parameters():
            p.requires_grad = False

        # Policy: (z, g) → action (tanh squashed to [-1, 1])
        self.policy_net = _mlp(z + g, h, a)

    # ──────────────────── Forward helpers ────────────────────

    def encode(self, state):
        """state [B, state_dim] → z [B, latent_dim]"""
        return self.encoder(state)

    def next_latent(self, z, action, goal):
        """(z, a, g) → z'"""
        return self.dynamics(torch.cat([z, action, goal], dim=-1))

    def predict_reward(self, z, action, goal):
        """(z, a, g) → r̂  [B, 1]"""
        return self.reward(torch.cat([z, action, goal], dim=-1))

    def predict_value(self, z, goal, use_target=False):
        """(z, g) → v̂  [B, 1]"""
        net = self.value_target if use_target else self.value
        return net(torch.cat([z, goal], dim=-1))

    def policy(self, z, goal):
        """(z, g) → a [B, action_dim]  (tanh squashed)"""
        return torch.tanh(self.policy_net(torch.cat([z, goal], dim=-1)))

    # ──────────────────── Training ────────────────────

    def compute_loss(self, states, actions, rewards, goals):
        """
        Compute all TD-MPC2 losses on a batch of trajectories.

        Args:
            states:  [B, T+1, state_dim]   (T transitions → T+1 states)
            actions: [B, T, action_dim]
            rewards: [B, T]
            goals:   [B, goal_dim]

        Returns:  dict of scalar losses
        """
        B, T_plus_1, _ = states.shape
        T = T_plus_1 - 1

        # Encode ALL timesteps at once
        z_all = self.encode(states.reshape(B * T_plus_1, -1)).reshape(B, T_plus_1, -1)
        z_targets = z_all[:, 1:].detach()  # stop gradient for consistency targets

        loss_dyn = 0.0
        loss_rew = 0.0
        loss_val = 0.0
        loss_pi = 0.0

        z = z_all[:, 0]  # start from encoded first state

        for t in range(T):
            a_t = actions[:, t]
            r_t = rewards[:, t]
            g = goals

            # ── Dynamics loss: predicted z' should match encoded s' ──
            z_pred = self.next_latent(z, a_t, g)
            loss_dyn += ((z_pred - z_targets[:, t]) ** 2).mean()

            # ── Reward loss ──
            r_pred = self.predict_reward(z, a_t, g).squeeze(-1)
            loss_rew += ((r_pred - r_t) ** 2).mean()

            # ── Value loss (TD target: r + γ * V_target(z')) ──
            with torch.no_grad():
                v_target = r_t + self.cfg.gamma * self.predict_value(
                    z_targets[:, t], g, use_target=True
                ).squeeze(-1)
            v_pred = self.predict_value(z, g).squeeze(-1)
            loss_val += ((v_pred - v_target) ** 2).mean()

            # ── Policy loss: maximize value of next state under policy ──
            a_pi = self.policy(z.detach(), g)
            z_pi = self.next_latent(z.detach(), a_pi, g)
            loss_pi += -self.predict_value(z_pi, g).mean()

            # Roll forward in latent space (detach to prevent gradient explosion)
            z = z_pred.detach()

        n = max(T, 1)
        return {
            "dynamics": loss_dyn / n,
            "reward": loss_rew / n,
            "value": loss_val / n,
            "policy": loss_pi / n,
        }

    @torch.no_grad()
    def update_target(self, tau):
        """Exponential moving average of value → value_target."""
        for p, p_tgt in zip(self.value.parameters(), self.value_target.parameters()):
            p_tgt.data.lerp_(p.data, tau)
