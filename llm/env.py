"""UR5e VLA Environment — language-conditioned reach with camera observations."""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class UR5eVLAEnv(gym.Env):
    """
    Multi-modal Gymnasium environment for UR5e reach tasks.

    Observation (Dict):
        image       – (3, H, W) uint8 from workspace camera
        state       – (15,) float32: qpos(6) + qvel(6) + ee_pos(3)
        instruction – (D,) float32: language embedding of the current task

    Action: (6,) float32 in [-1, 1] → position offsets from home
    Reward: −distance  (+10 if distance < threshold)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None, render_mode=None, task_embeddings=None):
        super().__init__()
        from vla.config import Config

        self.cfg = config or Config()
        self.render_mode = render_mode

        # ── MuJoCo model ──
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model, self.cfg.image_size, self.cfg.image_size
        )

        # ── MuJoCo IDs ──
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_cam"
        )
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._target_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for _, name in self.cfg.tasks
        }

        # ── Language embeddings (precomputed externally to save memory) ──
        self._task_embeddings = task_embeddings or {}

        # ── Observation & action spaces ──
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                0, 255,
                (3, self.cfg.image_size, self.cfg.image_size),
                dtype=np.uint8,
            ),
            "state": spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32),
            "instruction": spaces.Box(
                -np.inf, np.inf, (self.cfg.language_dim,), dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        # ── Internal state ──
        self._home = np.array(self.cfg.home_position, dtype=np.float64)
        self._task_instruction = None
        self._task_target = None
        self._task_emb = np.zeros(self.cfg.language_dim, dtype=np.float32)
        self._steps = 0
        self._viewer = None

    # ──────────────────────── helpers ────────────────────────

    def _ee_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _target_pos(self):
        return self.data.xpos[self._target_body_ids[self._task_target]].copy()

    def _render_camera(self):
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render()  # (H, W, 3) uint8

    def _get_obs(self):
        img = self._render_camera()
        img = np.transpose(img, (2, 0, 1)).copy()  # channels-first for CNN

        qpos = self.data.qpos[:6].astype(np.float32)
        qvel = self.data.qvel[:6].astype(np.float32)
        ee = self._ee_pos().astype(np.float32)
        state = np.concatenate([qpos, qvel, ee])

        return {
            "image": img,
            "state": state,
            "instruction": self._task_emb.copy(),
        }

    def _reward_and_info(self):
        ee = self._ee_pos()
        target = self._target_pos()
        dist = float(np.linalg.norm(ee - target))
        success = dist < self.cfg.success_threshold
        reward = -dist + (10.0 if success else 0.0)
        return reward, dist, success

    # ──────────────────────── Gymnasium API ────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo physics
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.normal(0, 0.02, size=6)
        self.data.qpos[:6] = self._home + noise
        self.data.ctrl[:6] = self._home
        mujoco.mj_forward(self.model, self.data)

        # Sample a random task for this episode
        idx = int(self.np_random.integers(0, len(self.cfg.tasks)))
        self._task_instruction, self._task_target = self.cfg.tasks[idx]
        self._task_emb = self._task_embeddings.get(
            self._task_instruction,
            np.zeros(self.cfg.language_dim, dtype=np.float32),
        )
        self._steps = 0

        info = {
            "instruction": self._task_instruction,
            "target": self._task_target,
        }
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Position control: home + action offset
        self.data.ctrl[:6] = self._home + action * self.cfg.action_scale

        # 5 physics sub-steps for stability
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        # Safety: NaN check → truncate episode
        if not np.all(np.isfinite(self.data.qpos[:6])):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:6] = self._home
            mujoco.mj_forward(self.model, self.data)
            return self._get_obs(), -10.0, False, True, {
                "instruction": self._task_instruction,
                "target": self._task_target,
                "distance": -1.0,
                "success": False,
            }

        reward, dist, success = self._reward_and_info()
        terminated = success
        truncated = self._steps >= self.cfg.max_episode_steps

        info = {
            "instruction": self._task_instruction,
            "target": self._task_target,
            "distance": dist,
            "success": success,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer.close()
