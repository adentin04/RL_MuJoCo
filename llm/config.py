"""Central configuration for hierarchical VLA + TD-MPC2 training."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Environment ──
    xml_path: str = "universal_robots_ur5e/scene_vla.xml"
    image_size: int = 64
    max_episode_steps: int = 200
    action_scale: float = 0.5
    action_dim: int = 6
    state_dim: int = 15   # qpos(6) + qvel(6) + ee(3)
    success_threshold: float = 0.05  # metres

    # ── Tasks: (instruction, target_body_name) ──
    tasks: list = field(default_factory=lambda: [
        ("reach the red target", "target_red"),
        ("reach the green target", "target_green"),
        ("reach the blue target", "target_blue"),
    ])

    # ── High-level: VLA goal encoder ──
    language_model: str = "all-MiniLM-L6-v2"
    language_dim: int = 384
    goal_dim: int = 64  # compressed goal vector sent to world model

    # ── Low-level: TD-MPC2 world model ──
    latent_dim: int = 128          # z dimension
    hidden_dim: int = 256          # MLP hidden layers
    horizon: int = 5               # planning horizon (imagined steps)
    num_samples: int = 512         # MPPI trajectories
    num_elites: int = 64           # top-k for MPPI
    temperature: float = 0.5       # MPPI softmax temperature
    momentum: float = 0.1          # MPPI action momentum

    # ── Training ──
    total_episodes: int = 2000
    seed_episodes: int = 5         # random exploration before training
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005             # target network EMA
    grad_clip: float = 10.0
    seq_len: int = 16              # trajectory chunk length for training

    # ── Logging ──
    log_dir: str = "vla/logs"
    db_path: str = "vla/experience.db"
    save_freq: int = 50            # every N episodes
    eval_freq: int = 25

    # ── Robot ──
    home_position: list = field(default_factory=lambda: [
        -1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0
    ])
