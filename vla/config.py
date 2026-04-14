"""Central configuration for VLA training."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Environment ──
    xml_path: str = "universal_robots_ur5e/scene_vla.xml"
    image_size: int = 64
    max_episode_steps: int = 200
    action_scale: float = 0.5
    success_threshold: float = 0.05  # metres

    # ── Tasks: (instruction, target_body_name) ──
    tasks: list = field(default_factory=lambda: [
        ("reach the red target", "target_red"),
        ("reach the green target", "target_green"),
        ("reach the blue target", "target_blue"),
    ])

    # ── Language encoder ──
    language_model: str = "all-MiniLM-L6-v2"
    language_dim: int = 384

    # ── Training (SAC) ──
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005

    # ── Network architecture ──
    image_features: int = 128
    state_features: int = 64
    language_features: int = 64
    combined_features: int = 256

    # ── Logging ──
    log_dir: str = "vla/logs"
    db_path: str = "vla/experience.db"
    save_freq: int = 10_000
    eval_freq: int = 5_000

    # ── Robot ──
    home_position: list = field(default_factory=lambda: [
        -1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0
    ])
