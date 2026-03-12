"""UR5e Reach — environnement MuJoCo au format dm_env (Acme)."""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import dm_env
import mujoco
import mujoco.viewer
import numpy as np
from dm_env import specs

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclasses.dataclass
class UR5eState:
    qpos: np.ndarray   # positions articulaires (6,)
    qvel: np.ndarray   # vitesses articulaires  (6,)
    target: np.ndarray # position cible 3D      (3,)
    time: float


class UR5eReachEnvDM(dm_env.Environment):
    """Environnement dm_env pour UR5e reach task.

    Observation : qpos(6) + qvel(6) + vecteur_relatif(3) = 15
    Action      : delta joint positions normalisé [-1, 1], shape (6,)
    Reward      : -distance  (+10 si distance < 5 cm)
    """

    HOME = np.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)
    TARGET = np.array([0.35, -0.25, 0.59], dtype=np.float32)
    ACTION_SCALE = 0.5
    SUCCESS_THRESHOLD = 0.05  # mètres

    def __init__(
        self,
        xml_path: str = "universal_robots_ur5e/ur5e.xml",
        render_mode: Optional[str] = None,
    ):
        self._log(f"init start: xml_path={xml_path}, render_mode={render_mode}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self._ee_site_id = (
            self.model.site("attachment_site").id
            if "attachment_site" in site_names
            else None
        )

        self._action_spec = specs.BoundedArray(
            shape=(6,), dtype=np.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )
        self._observation_spec = specs.BoundedArray(
            shape=(15,), dtype=np.float32,
            minimum=-np.inf, maximum=np.inf, name="observation",
        )

        self._state: UR5eState = self._make_state()
        self._render_mode = render_mode
        self._viewer = None
        self._episode_index = 0
        self._step_in_episode = 0
        self._global_step = 0
        self._last_episode_return = 0.0
        self._log(
            "init done: "
            f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}, "
            f"obs_shape={self._observation_spec.shape}, action_shape={self._action_spec.shape}, "
            f"ee_site_id={self._ee_site_id}"
        )

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][ENV] {msg}", flush=True)

    # ------------------------------------------------------------------
    # dm_env interface
    # ------------------------------------------------------------------

    def reset(self) -> dm_env.TimeStep:
        self._episode_index += 1
        self._step_in_episode = 0
        self._last_episode_return = 0.0
        self._log(f"reset start: episode={self._episode_index}")
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.HOME + np.random.uniform(-0.1, 0.1, 6).astype(np.float32)
        self.data.qpos[:6] = qpos
        self.data.ctrl[:6] = qpos
        mujoco.mj_forward(self.model, self.data)
        self._state = self._make_state()
        ee_pos = self._ee_pos()
        distance = float(np.linalg.norm(ee_pos - self._state.target))
        self._log(
            f"reset done: episode={self._episode_index}, qpos={np.round(qpos, 3).tolist()}, "
            f"target={np.round(self._state.target, 3).tolist()}, ee={np.round(ee_pos, 3).tolist()}, "
            f"start_distance={distance:.4f}"
        )
        return dm_env.restart(self._get_obs())

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self._step_in_episode += 1
        self._global_step += 1
        raw_action = np.asarray(action, dtype=np.float32)
        action = np.clip(
            np.nan_to_num(raw_action), -1.0, 1.0
        )
        if np.any(~np.isfinite(raw_action)):
            self._log(
                f"step warning: non-finite action détectée ep={self._episode_index} step={self._step_in_episode}"
            )

        ctrl = self.HOME + action * self.ACTION_SCALE
        self.data.ctrl[:6] = ctrl

        if self._step_in_episode <= 3 or self._step_in_episode % 25 == 0:
            self._log(
                f"step input: ep={self._episode_index} step={self._step_in_episode} "
                f"action[min,max]=({float(np.min(action)):.3f},{float(np.max(action)):.3f}) "
                f"ctrl[min,max]=({float(np.min(ctrl)):.3f},{float(np.max(ctrl)):.3f})"
            )

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        if not (
            np.all(np.isfinite(self.data.qpos[:6]))
            and np.all(np.isfinite(self.data.qvel[:6]))
        ):
            self._log(
                f"step invalid state: ep={self._episode_index} step={self._step_in_episode} -> truncation(-10)"
            )
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return dm_env.truncation(reward=-10.0, observation=self._get_obs())

        self._state = UR5eState(
            qpos=self.data.qpos[:6].copy().astype(np.float32),
            qvel=self.data.qvel[:6].copy().astype(np.float32),
            target=self._state.target,
            time=float(self.data.time),
        )

        ee_pos = self._ee_pos()
        distance = float(np.linalg.norm(ee_pos - self._state.target))
        reward = -distance + (10.0 if distance < self.SUCCESS_THRESHOLD else 0.0)
        self._last_episode_return += reward

        if self._step_in_episode <= 3 or self._step_in_episode % 25 == 0 or distance < self.SUCCESS_THRESHOLD:
            self._log(
                f"step stats: ep={self._episode_index} step={self._step_in_episode} "
                f"time={self._state.time:.3f}s distance={distance:.4f} reward={reward:.4f} "
                f"return_acc={self._last_episode_return:.4f}"
            )

        if distance < self.SUCCESS_THRESHOLD:
            self._log(
                f"episode termination: SUCCESS ep={self._episode_index} step={self._step_in_episode} "
                f"distance={distance:.4f} return={self._last_episode_return:.4f}"
            )
            return dm_env.termination(reward=reward, observation=self._get_obs())
        if self._state.time > 60.0:
            self._log(
                f"episode truncation: TIMEOUT ep={self._episode_index} step={self._step_in_episode} "
                f"sim_time={self._state.time:.3f}s return={self._last_episode_return:.4f}"
            )
            return dm_env.truncation(reward=reward, observation=self._get_obs())
        return dm_env.transition(reward=reward, observation=self._get_obs())

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def observation_spec(self) -> specs.BoundedArray:
        return self._observation_spec

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        if self._render_mode not in {"human", "viewer"}:
            return
        if self._viewer is None:
            self._log("render: ouverture viewer passif")
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._log("close: fermeture viewer")
            self._viewer.close()
            self._viewer = None
        self._log("close: environnement fermé")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_state(self) -> UR5eState:
        return UR5eState(
            qpos=self.data.qpos[:6].copy().astype(np.float32),
            qvel=self.data.qvel[:6].copy().astype(np.float32),
            target=self.TARGET.copy(),
            time=0.0,
        )

    def _ee_pos(self) -> np.ndarray:
        if self._ee_site_id is not None:
            return self.data.site_xpos[self._ee_site_id].copy()
        return self.data.geom_xpos[-1].copy()

    def _get_obs(self) -> np.ndarray:
        relative = self._state.target - self._ee_pos()
        return np.concatenate([
            self._state.qpos,
            self._state.qvel,
            relative.astype(np.float32),
        ]).astype(np.float32)


# ------------------------------------------------------------------
# Utilitaire visualisation
# ------------------------------------------------------------------

def plot_training_returns(returns: list[float]) -> None:
    if plt is None:
        print("matplotlib non disponible.")
        return

    y = np.asarray(returns, dtype=np.float32)
    x = np.arange(len(y))
    window = min(10, len(y))
    avg = np.convolve(y, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, alpha=0.35, label="Return épisode")
    plt.plot(np.arange(window - 1, len(y)), avg, linewidth=2, label=f"Moyenne mobile ({window})")
    plt.xlabel("Épisode")
    plt.ylabel("Return")
    plt.title("Progression entraînement UR5e")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
