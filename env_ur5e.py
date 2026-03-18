"""UR5e Reach — environnement MuJoCo au format dm_env (Acme)."""

from __future__ import annotations

import dataclasses
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
class UR5eState:   #variabileistato del braccio robotico 
    qpos: np.ndarray   # positions articulaires (6,)
    qvel: np.ndarray   # vitesses articulaires  (6,)
    target: np.ndarray # position cible 3D      (3,)
    time: float


class UR5eReachEnvDM(dm_env.Environment): #classe stato del braccio robotico
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
        self, #this
        xml_path: str = "universal_robots_ur5e/ur5e.xml", # il robot
        render_mode: Optional[str] = None, #per il viewer
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path) #variabile per il modello 
        self.data = mujoco.MjData(self.model) #i dati importanti sul robot

        site_names = [self.model.site(i).name for i in range(self.model.nsite)] #?
        self._ee_site_id = ( #?
            self.model.site("attachment_site").id
            if "attachment_site" in site_names
            else None
        )

        self._action_spec = specs.BoundedArray( #?
            shape=(6,), dtype=np.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )
        self._observation_spec = specs.BoundedArray( #?
            shape=(15,), dtype=np.float32,
            minimum=-np.inf, maximum=np.inf, name="observation",
        )

        self._state: UR5eState = self._make_state()
        self._render_mode = render_mode
        self._viewer = None

    # ------------------------------------------------------------------
    # dm_env interface
    # ------------------------------------------------------------------

    def reset(self) -> dm_env.TimeStep:
        mujoco.mj_resetData(self.model, self.data)# reset allo stato iniziale 
        qpos = self.HOME + np.random.uniform(-0.1, 0.1, 6).astype(np.float32)
        self.data.qpos[:6] = qpos
        self.data.ctrl[:6] = qpos
        mujoco.mj_forward(self.model, self.data)
        self._state = self._make_state()
        return dm_env.restart(self._get_obs())

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        action = np.clip(
            np.nan_to_num(np.asarray(action, dtype=np.float32)), -1.0, 1.0
        )
        ctrl = self.HOME + action * self.ACTION_SCALE
        self.data.ctrl[:6] = ctrl

        for _ in range(5):
            mujoco.mj_step(self.model, self.data) #?

        if not (
            np.all(np.isfinite(self.data.qpos[:6]))
            and np.all(np.isfinite(self.data.qvel[:6]))
        ):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return dm_env.truncation(reward=-30.0, observation=self._get_obs())

        self._state = UR5eState(
            qpos=self.data.qpos[:6].copy().astype(np.float32),
            qvel=self.data.qvel[:6].copy().astype(np.float32),
            target=self._state.target,
            time=float(self.data.time),
        )

        ee_pos = self._ee_pos()
        distance = float(np.linalg.norm(ee_pos - self._state.target))
        reward = -distance + (10.0 if distance < self.SUCCESS_THRESHOLD else 0.0)

        if distance < self.SUCCESS_THRESHOLD:
            return dm_env.termination(reward=reward, observation=self._get_obs())
        if self._state.time > 60.0:
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
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

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
print("dm_env Trunc", dm_env.truncation)
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
