"""
Train the hierarchical VLA + TD-MPC2 agent.

High-level:  VLA goal encoder  (image + language → goal)
Low-level:   TD-MPC2 world model + MPPI planner  (state + goal → action)
"""

import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from vla.config import Config
from vla.env import UR5eVLAEnv
from vla.experience_db import ExperienceDB
from vla.goal_encoder import GoalEncoder
from vla.planner import MPPIPlanner
from vla.replay_buffer import ReplayBuffer
from vla.world_model import WorldModel


def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# ────────────────────────── Training step ──────────────────────────


def train_step(world_model, goal_encoder, optimizer, buffer, cfg, device):
    """One gradient step on both networks using a batch from the buffer."""
    states, actions, rewards, goals_np = buffer.sample_chunks(
        cfg.batch_size, cfg.seq_len
    )

    states_t = _to_tensor(states, device)
    actions_t = _to_tensor(actions, device)
    rewards_t = _to_tensor(rewards, device)
    goals_t = _to_tensor(goals_np, device)

    # World model losses
    losses = world_model.compute_loss(states_t, actions_t, rewards_t, goals_t)
    total_loss = losses["dynamics"] + losses["reward"] + losses["value"] + 0.1 * losses["policy"]

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(world_model.parameters()) + list(goal_encoder.parameters()),
        cfg.grad_clip,
    )
    optimizer.step()

    # Soft update target value network
    world_model.update_target(cfg.tau)

    return {k: v.item() for k, v in losses.items()}


# ────────────────────────── Episode rollout ──────────────────────────


def run_episode(env, world_model, goal_encoder, planner, buffer, cfg, device,
                task_embeddings, explore=False):
    """
    Run one episode. Returns (total_reward, steps, success, distance, instruction).
    """
    obs, info = env.reset()
    instruction = info["instruction"]

    # High-level: compute goal from image + language
    img_t = _to_tensor(obs["image"], device).unsqueeze(0).float() / 255.0
    lang_t = _to_tensor(task_embeddings[instruction], device).unsqueeze(0)
    goal = goal_encoder(img_t, lang_t).squeeze(0)  # [goal_dim]

    planner.reset()
    total_reward = 0.0
    steps = 0
    success = False
    distance = -1.0

    for _ in range(cfg.max_episode_steps):
        state = obs["state"]

        if explore:
            action = env.action_space.sample()
        else:
            action = planner.plan(state, goal, device=device)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store transition
        buffer.add(
            state=state,
            action=action,
            reward=reward,
            goal=goal.detach().cpu().numpy(),
            done=terminated or truncated,
        )

        total_reward += reward
        steps += 1
        distance = info.get("distance", -1.0)
        success = info.get("success", False)
        obs = next_obs

        if terminated or truncated:
            break

    return total_reward, steps, success, distance, instruction


# ──────────────────────── Entry point ─────────────────────────


def train(cfg: Config = None):
    cfg = cfg or Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VLA] Device: {device}")

    # 1. Language encoder — encode task instructions once
    print("[VLA] Loading language model …")
    encoder = SentenceTransformer(cfg.language_model)
    task_embeddings = {
        instr: encoder.encode(instr).astype(np.float32)
        for instr, _ in cfg.tasks
    }
    del encoder

    # 2. Environment
    print("[VLA] Creating environment …")
    env = UR5eVLAEnv(config=cfg, task_embeddings=task_embeddings)

    # 3. Models
    print("[VLA] Building models …")
    goal_enc = GoalEncoder(cfg).to(device)
    world_model = WorldModel(cfg).to(device)
    planner = MPPIPlanner(cfg, world_model)

    all_params = list(world_model.parameters()) + list(goal_enc.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate)

    # 4. Replay buffer & experience DB
    buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim, cfg.action_dim, cfg.goal_dim)
    db = ExperienceDB(cfg.db_path)

    # 5. Seed episodes (random exploration)
    print(f"[VLA] Collecting {cfg.seed_episodes} seed episodes …")
    for ep in range(cfg.seed_episodes):
        ret, steps, suc, dist, instr = run_episode(
            env, world_model, goal_enc, planner, buffer, cfg, device,
            task_embeddings, explore=True,
        )
        db.log_episode(instr, "", ret, steps, suc, dist)
        print(f"  Seed {ep+1}/{cfg.seed_episodes}: reward={ret:.1f}, dist={dist:.3f}")

    # 6. Main training loop
    print(f"[VLA] Training for {cfg.total_episodes} episodes …")
    for ep in range(1, cfg.total_episodes + 1):
        # Collect one episode with MPPI planning
        ret, steps, suc, dist, instr = run_episode(
            env, world_model, goal_enc, planner, buffer, cfg, device,
            task_embeddings, explore=False,
        )
        db.log_episode(instr, "", ret, steps, suc, dist)

        # Train on buffer (multiple gradient steps per episode)
        if buffer.size > cfg.seq_len + 2:
            n_updates = max(1, steps // cfg.seq_len)
            for _ in range(n_updates):
                try:
                    losses = train_step(world_model, goal_enc, optimizer, buffer, cfg, device)
                except ValueError:
                    break

        # Logging
        if ep % 10 == 0:
            stats = db.stats()
            print(
                f"  Episode {ep:4d} | reward={ret:7.1f} | dist={dist:.3f} | "
                f"success={'YES' if suc else 'no ':3s} | task={instr} | "
                f"avg_success={stats['success_rate']:.1%}"
            )

        # Save checkpoint
        if ep % cfg.save_freq == 0:
            path = os.path.join(cfg.log_dir, f"checkpoint_{ep}.pt")
            torch.save({
                "world_model": world_model.state_dict(),
                "goal_encoder": goal_enc.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep,
            }, path)
            print(f"  [SAVE] {path}")

    # 7. Final save
    path = os.path.join(cfg.log_dir, "final.pt")
    torch.save({
        "world_model": world_model.state_dict(),
        "goal_encoder": goal_enc.state_dict(),
    }, path)
    print(f"\n[VLA] Final model → {path}")

    # 8. Summary
    print("\n── Training summary ─────────────────────")
    print(f"  Overall: {db.stats()}")
    for instr, _ in cfg.tasks:
        print(f"  {instr}: {db.stats(instr)}")

    db.close()
    env.close()


if __name__ == "__main__":
    train()
