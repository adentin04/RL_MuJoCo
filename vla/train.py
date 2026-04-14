"""Train the VLA agent (SAC) on language-conditioned UR5e reach tasks."""

import os

import numpy as np
from sentence_transformers import SentenceTransformer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from vla.config import Config
from vla.env import UR5eVLAEnv
from vla.experience_db import ExperienceDB
from vla.policy import VLAExtractor


# ────────────────────────── Callback ──────────────────────────


class DBLoggerCallback(BaseCallback):
    """Logs completed episodes to the experience database."""

    def __init__(self, db: ExperienceDB, verbose=0):
        super().__init__(verbose)
        self.db = db

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "episode" in info:
                self.db.log_episode(
                    instruction=info.get("instruction", ""),
                    target=info.get("target", ""),
                    total_reward=info["episode"]["r"],
                    steps=int(info["episode"]["l"]),
                    success=info.get("success", False),
                    final_distance=info.get("distance", -1.0),
                )
        return True


# ──────────────────────── Entry point ─────────────────────────


def train(cfg: Config = None):
    cfg = cfg or Config()
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 1. Language encoder — encode task instructions once, then free memory
    print("[VLA] Loading language model …")
    encoder = SentenceTransformer(cfg.language_model)
    task_embeddings = {
        instr: encoder.encode(instr).astype(np.float32)
        for instr, _ in cfg.tasks
    }
    del encoder  # free ~80 MB

    # 2. Environment
    print("[VLA] Creating environment …")
    env = UR5eVLAEnv(config=cfg, task_embeddings=task_embeddings)
    env = Monitor(env)

    # 3. Experience database
    db = ExperienceDB(cfg.db_path)

    # 4. SAC with custom VLA feature extractor
    print("[VLA] Building SAC agent …")
    agent = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": VLAExtractor,
            "features_extractor_kwargs": {"config": cfg},
            "net_arch": [256, 256],
        },
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        tensorboard_log=cfg.log_dir,
        verbose=1,
    )

    # 5. Train
    print(f"[VLA] Training for {cfg.total_timesteps} steps …")
    agent.learn(
        total_timesteps=cfg.total_timesteps,
        callback=DBLoggerCallback(db),
        log_interval=10,
    )

    # 6. Save model
    save_path = os.path.join(cfg.log_dir, "vla_final")
    agent.save(save_path)
    print(f"[VLA] Model saved → {save_path}")

    # 7. Print summary
    print("\n── Training summary ─────────────────────")
    print(f"  Overall: {db.stats()}")
    for instr, _ in cfg.tasks:
        print(f"  {instr}: {db.stats(instr)}")

    db.close()
    env.close()


if __name__ == "__main__":
    train()
