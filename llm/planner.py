"""
MPPI Planner — Model Predictive Path Integral.

Uses the learned world model to imagine many possible futures,
then picks the best action sequence.

Algorithm:
    1. Sample N action trajectories (warm-started from learned policy + noise)
    2. Roll each trajectory through the world model in latent space
    3. Score each trajectory: sum of predicted rewards + terminal value
    4. Weight trajectories by softmax(scores / temperature)
    5. Return weighted mean of first actions
"""

import torch


class MPPIPlanner:
    def __init__(self, cfg, world_model):
        self.cfg = cfg
        self.wm = world_model
        self._prev_mean = None  # action momentum across planning steps

    @torch.no_grad()
    def plan(self, state, goal, device="cpu"):
        """
        Run MPPI planning from a single state.

        Args:
            state: [state_dim] numpy or tensor — current robot state
            goal:  [goal_dim] tensor — goal from VLA encoder
            device: torch device

        Returns:
            action: [action_dim] numpy — best first action
        """
        cfg = self.cfg
        H = cfg.horizon
        N = cfg.num_samples
        K = cfg.num_elites
        a_dim = cfg.action_dim

        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        goal_t = goal.to(device) if isinstance(goal, torch.Tensor) else torch.as_tensor(goal, dtype=torch.float32, device=device)

        # Encode current state → latent
        z0 = self.wm.encode(state_t.unsqueeze(0))  # [1, latent_dim]
        z_batch = z0.expand(N, -1)                  # [N, latent_dim]
        g_batch = goal_t.unsqueeze(0).expand(N, -1) # [N, goal_dim]

        # ── Initialize action sequences from learned policy + noise ──
        # Get policy's guess at each step (will be refined by MPPI)
        mean = torch.zeros(H, a_dim, device=device)
        z_tmp = z_batch[0:1]
        for t in range(H):
            a_pi = self.wm.policy(z_tmp, goal_t.unsqueeze(0))
            mean[t] = a_pi.squeeze(0)
            z_tmp = self.wm.next_latent(z_tmp, a_pi, goal_t.unsqueeze(0))

        # Apply momentum from previous planning step
        if self._prev_mean is not None:
            mean = cfg.momentum * mean + (1 - cfg.momentum) * torch.cat(
                [self._prev_mean[1:], mean[-1:]], dim=0
            )

        # ── Sample N trajectories around the mean ──
        noise = 0.5 * torch.randn(N, H, a_dim, device=device)
        actions = (mean.unsqueeze(0) + noise).clamp(-1.0, 1.0)  # [N, H, a_dim]

        # ── Evaluate trajectories in latent space ──
        total_rewards = torch.zeros(N, device=device)
        z = z_batch

        for t in range(H):
            a_t = actions[:, t]
            r_t = self.wm.predict_reward(z, a_t, g_batch).squeeze(-1)
            total_rewards += (cfg.gamma ** t) * r_t
            z = self.wm.next_latent(z, a_t, g_batch)

        # Add terminal value
        v_terminal = self.wm.predict_value(z, g_batch, use_target=True).squeeze(-1)
        total_rewards += (cfg.gamma ** H) * v_terminal

        # ── Select elites and compute weighted mean ──
        _, elite_idx = total_rewards.topk(K)
        elite_actions = actions[elite_idx]  # [K, H, a_dim]
        elite_rewards = total_rewards[elite_idx]

        # Softmax weighting
        weights = torch.softmax(elite_rewards / cfg.temperature, dim=0)  # [K]
        plan = (weights.unsqueeze(-1).unsqueeze(-1) * elite_actions).sum(dim=0)  # [H, a_dim]

        # Save for momentum
        self._prev_mean = plan.clone()

        # Return first action
        return plan[0].cpu().numpy()

    def reset(self):
        """Reset momentum when starting a new episode."""
        self._prev_mean = None
