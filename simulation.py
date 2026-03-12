"""UR5e Reach Environment avec agent JAX local (sans Acme)."""

import dataclasses
from typing import Optional, Dict, Any, Tuple

import dm_env
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from dm_env import specs

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

@dataclasses.dataclass
class UR5eState:
    """État complet pour JAX (pytree)"""
    qpos: jnp.ndarray      # positions articulaires (6,)
    qvel: jnp.ndarray      # vitesses (6,)
    target: jnp.ndarray     # position cible (3,)
    time: float
    last_distance: float
    curriculum_stage: int

class UR5eReachEnvDM(dm_env.Environment):
    """
    Environnement au format DeepMind (dm_env) pour Acme
    """
    
    def __init__(self, 
                 xml_path: str = 'universal_robots_ur5e/ur5e.xml',
                 render_mode: Optional[str] = None):
        # Initialisation MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Paramètres du robot
        self.home_joints = jnp.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.action_scale = 0.5
        
        # Curriculum targets
        self.curriculum_targets = [
            jnp.array([0.26, -0.34, 0.59]),
            jnp.array([0.35, -0.25, 0.59]),
            jnp.array([0.41, -0.14, 0.59]),
        ]
        
        # Spécifications dm_env (équivalent gym.space)
        self._action_spec = specs.BoundedArray(
            shape=(6,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
        
        # Observation: qpos(6) + qvel(6) + ee_pos(3) + target(3) + relative(3) = 21
        self._observation_spec = specs.BoundedArray(
            shape=(21,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='observation'
        )
        
        # ID du site effecteur
        site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self.ee_site_id = self.model.site('attachment_site').id if 'attachment_site' in site_names else None
        
        # État JAX (initial)
        self._state = self._initialize_state()
        
        # Pour le rendering
        self._render_mode = render_mode
        self._viewer = None
        self._viewer_enabled = render_mode in {"human", "viewer"}
        
    def _initialize_state(self) -> UR5eState:
        """Crée l'état initial"""
        ee_pos = self._get_end_effector_pos()
        initial_distance = float(np.linalg.norm(ee_pos - np.array(self.curriculum_targets[0])))
        return UR5eState(
            qpos=jnp.array(self.data.qpos[:6].copy()),
            qvel=jnp.array(self.data.qvel[:6].copy()),
            target=self.curriculum_targets[0],
            time=0.0,
            last_distance=initial_distance,
            curriculum_stage=0
        )
    
    def reset(self) -> dm_env.TimeStep:
        """Reset dm_env style"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale avec bruit
        initial_qpos = self.home_joints + np.random.uniform(-0.1, 0.1, 6)
        self.data.qpos[:6] = initial_qpos
        self.data.ctrl[:6] = initial_qpos
        
   

        ee_pos = self._get_end_effector_pos()
        initial_distance = float(np.linalg.norm(ee_pos - np.array(self.curriculum_targets[0])))
        
        # Mettre à jour l'état JAX
        self._state = UR5eState(
            qpos=jnp.array(initial_qpos),
            qvel=jnp.zeros(6),
            target=self.curriculum_targets[0],
            time=0.0,
            last_distance=initial_distance,
            curriculum_stage=0
        )
        
        # Premier timestep dm_env
        return dm_env.restart(self._get_obs())
    
    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Step dm_env style"""
        safe_action = np.asarray(action, dtype=np.float32)
        safe_action = np.nan_to_num(safe_action, nan=0.0, posinf=1.0, neginf=-1.0)
        safe_action = np.clip(safe_action, -1.0, 1.0)

        ctrl = np.asarray(self.home_joints) + safe_action * self.action_scale
        ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.model.nu >= 6 and getattr(self.model, "actuator_ctrllimited", None) is not None:
            limited = np.asarray(self.model.actuator_ctrllimited[:6]).astype(bool)
            if np.any(limited):
                low = np.asarray(self.model.actuator_ctrlrange[:6, 0])
                high = np.asarray(self.model.actuator_ctrlrange[:6, 1])
                ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])

        self.data.ctrl[:6] = ctrl
        
        # Simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        if (not np.all(np.isfinite(self.data.qpos[:6])) or
                not np.all(np.isfinite(self.data.qvel[:6])) or
                not np.all(np.isfinite(self.data.ctrl[:6]))):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return dm_env.truncation(reward=-10.0, observation=self._get_obs())
        
        # Nouvel état
        new_qpos = jnp.array(self.data.qpos[:6].copy())
        new_qvel = jnp.array(self.data.qvel[:6].copy())
        new_time = self.data.time
        
        # Récompense
        reward = self._compute_reward()
        
        # Vérifier terminaison
        ee_pos = self._get_end_effector_pos()
        distance = float(jnp.linalg.norm(ee_pos - self._state.target))
        terminated = distance < 0.05
        truncated = new_time > 60.0  # 60s sim time ≈ 6000 steps
        
        # Mettre à jour l'état JAX
        self._state = UR5eState(
            qpos=new_qpos,
            qvel=new_qvel,
            target=self._state.target,
            time=new_time,
            last_distance=distance,
            curriculum_stage=self._state.curriculum_stage
        )
        
        # Curriculum learning
        if terminated:
            self._state = self._update_curriculum(self._state)
        
        # Retour dm_env
        if terminated:
            return dm_env.termination(reward=reward, observation=self._get_obs())
        elif truncated:
            return dm_env.truncation(reward=reward, observation=self._get_obs())
        else:
            return dm_env.transition(reward=reward, observation=self._get_obs())
    
    def _get_obs(self) -> np.ndarray:
        """Observation numpy pour l'interface"""
        ee_pos = self._get_end_effector_pos()
        relative = self._state.target - ee_pos
        
        obs = np.concatenate([
            np.array(self._state.qpos),
            np.array(self._state.qvel),
            np.array(ee_pos),
            np.array(self._state.target),
            np.array(relative)
        ]).astype(np.float32)
        return obs
    
    def _get_obs_jax(self) -> jnp.ndarray:
        """Version JAX pour les networks"""
        ee_pos = self._get_end_effector_pos_jax()
        relative = self._state.target - ee_pos
        
        return jnp.concatenate([
            self._state.qpos,
            self._state.qvel,
            ee_pos,
            self._state.target,
            relative
        ])
    
    def _get_end_effector_pos(self) -> np.ndarray:
        """Position effecteur en numpy"""
        if self.ee_site_id is not None:
            return self.data.site_xpos[self.ee_site_id].copy()
        return self.data.geom_xpos[-1].copy()
    

    def _compute_reward(self) -> float:
        """Calcul récompense"""
        ee_pos = self._get_end_effector_pos()
        distance = float(np.linalg.norm(ee_pos - np.array(self._state.target)))
        distance = float(np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0))

        # Base positive pour éviter des retours systématiquement négatifs
        step_bonus = 0.25

        # Récompense principale (proximité)
        distance_reward = 1.0 - np.tanh(3.0 * distance)

        # Progression locale
        progress = float(self._state.last_distance - distance)
        progress = float(np.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0))
        progress = float(np.clip(progress, -1.0, 1.0))
        progress_reward = 2.0 * progress

        # Alignement vertical léger
        z_penalty = -0.3 * abs(ee_pos[2] - float(self._state.target[2]))

        # Pénalité douceur
        smoothness = -0.001 * float(np.mean(np.abs(self.data.ctrl[:6])))

        # Bonus succès fort
        success = 20.0 if distance < 0.05 else 0.0

        reward = float(step_bonus + distance_reward + progress_reward + z_penalty + smoothness + success)
        return float(np.nan_to_num(reward, nan=-1.0, posinf=5.0, neginf=-5.0))
    
    def _update_curriculum(self, state: UR5eState) -> UR5eState:
        """Mise à jour curriculum"""
        # Logique curriculum...
        return state
    
    # Spécifications dm_env requises
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def render(self, mode='human'):
        """Rendu MuJoCo en viewer passif."""
        if not self._viewer_enabled:
            return

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
    
    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None


@dataclasses.dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, transition: Transition):
        i = self.index
        self.obs[i] = np.nan_to_num(transition.obs, nan=0.0, posinf=1e3, neginf=-1e3)
        self.actions[i] = np.nan_to_num(transition.action, nan=0.0, posinf=1.0, neginf=-1.0)
        self.rewards[i] = float(np.nan_to_num(transition.reward, nan=0.0, posinf=10.0, neginf=-10.0))
        self.next_obs[i] = np.nan_to_num(transition.next_obs, nan=0.0, posinf=1e3, neginf=-1e3)
        self.dones[i] = float(np.nan_to_num(transition.done, nan=1.0, posinf=1.0, neginf=1.0))
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


def init_linear_params(key, in_dim: int, out_dim: int, scale: float = 0.1):
    w_key, _ = jax.random.split(key)
    w = scale * jax.random.normal(w_key, shape=(in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def linear(params, x):
    return x @ params["w"] + params["b"]


def init_actor_params(key, obs_dim: int, act_dim: int):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "l1": init_linear_params(k1, obs_dim, 128),
        "l2": init_linear_params(k2, 128, 128),
        "out": init_linear_params(k3, 128, act_dim, scale=0.01),
    }


def init_critic_params(key, obs_dim: int, act_dim: int):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "l1": init_linear_params(k1, obs_dim + act_dim, 128),
        "l2": init_linear_params(k2, 128, 128),
        "out": init_linear_params(k3, 128, 1, scale=0.01),
    }


def actor_forward(params, obs):
    x = jnp.tanh(linear(params["l1"], obs))
    x = jnp.tanh(linear(params["l2"], x))
    return jnp.tanh(linear(params["out"], x))


def critic_forward(params, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    x = jnp.tanh(linear(params["l1"], x))
    x = jnp.tanh(linear(params["l2"], x))
    return linear(params["out"], x).squeeze(-1)


def tree_sgd_step(params, grads, lr: float):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


class JAXReachAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        batch_size: int = 256,
        min_replay_size: int = 5000,
        replay_capacity: int = 200000,
        noise_std: float = 0.15,
    ):
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.noise_std = noise_std

        self.np_rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        key_actor, key_critic = jax.random.split(key)

        self.actor_params = init_actor_params(key_actor, obs_dim, action_dim)
        self.critic_params = init_critic_params(key_critic, obs_dim, action_dim)

        self.target_actor_params = self.actor_params
        self.target_critic_params = self.critic_params

        self.replay = ReplayBuffer(replay_capacity, obs_dim, action_dim)

        self._update_jit = jax.jit(self._update_step)

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        safe_obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1e3, neginf=-1e3)
        obs_j = jnp.asarray(safe_obs)
        action = np.asarray(actor_forward(self.actor_params, obs_j), dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        if explore:
            action = action + self.np_rng.normal(0.0, self.noise_std, size=action.shape).astype(np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, done):
        self.replay.add(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=float(done),
            )
        )

    def should_update(self) -> bool:
        return self.replay.size >= self.min_replay_size

    def update(self):
        if not self.should_update():
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = self.replay.sample(self.batch_size, self.np_rng)
        obs, actions, rewards, next_obs, dones = [jnp.asarray(x) for x in batch]
        if not (
            np.all(np.isfinite(np.asarray(obs)))
            and np.all(np.isfinite(np.asarray(actions)))
            and np.all(np.isfinite(np.asarray(rewards)))
            and np.all(np.isfinite(np.asarray(next_obs)))
            and np.all(np.isfinite(np.asarray(dones)))
        ):
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        (
            self.actor_params,
            self.critic_params,
            self.target_actor_params,
            self.target_critic_params,
            critic_loss,
            actor_loss,
        ) = self._update_jit(
            self.actor_params,
            self.critic_params,
            self.target_actor_params,
            self.target_critic_params,
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            self.gamma,
            self.actor_lr,
            self.critic_lr,
            self.tau,
        )

        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss)}

    @staticmethod
    def _update_step(
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        gamma,
        actor_lr,
        critic_lr,
        tau,
    ):
        next_actions = jax.vmap(lambda o: actor_forward(target_actor_params, o))(next_obs)
        target_q = jax.vmap(lambda o, a: critic_forward(target_critic_params, o, a))(next_obs, next_actions)
        y = rewards + gamma * (1.0 - dones) * target_q
        y = jnp.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0)

        def critic_loss_fn(c_params):
            q = jax.vmap(lambda o, a: critic_forward(c_params, o, a))(obs, actions)
            q = jnp.nan_to_num(q, nan=0.0, posinf=100.0, neginf=-100.0)
            return jnp.mean((q - y) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_params = tree_sgd_step(critic_params, critic_grads, critic_lr)

        def actor_loss_fn(a_params):
            pred_actions = jax.vmap(lambda o: actor_forward(a_params, o))(obs)
            pred_actions = jnp.nan_to_num(pred_actions, nan=0.0, posinf=1.0, neginf=-1.0)
            q_pred = jax.vmap(lambda o, a: critic_forward(critic_params, o, a))(obs, pred_actions)
            q_pred = jnp.nan_to_num(q_pred, nan=0.0, posinf=100.0, neginf=-100.0)
            return -jnp.mean(q_pred)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_params = tree_sgd_step(actor_params, actor_grads, actor_lr)

        target_actor_params = jax.tree_util.tree_map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_actor_params,
            actor_params,
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_critic_params,
            critic_params,
        )

        return (
            actor_params,
            critic_params,
            target_actor_params,
            target_critic_params,
            critic_loss,
            actor_loss,
        )


def train_ur5e_acme():
    """Entraînement avec agent JAX local (remplacement d'Acme)."""

    environment = UR5eReachEnvDM(render_mode='human')

    obs_dim = int(environment.observation_spec().shape[0])
    act_dim = int(environment.action_spec().shape[0])
    agent = JAXReachAgent(obs_dim=obs_dim, action_dim=act_dim)

    num_episodes = 100000
    max_steps_per_episode = 500

    episode_returns = []
    episode_success = []

    for episode in range(num_episodes):
        timestep = environment.reset()
        episode_return = 0.0

        for step in range(max_steps_per_episode):
            obs = timestep.observation
            action = agent.select_action(obs, explore=True)
            timestep = environment.step(action)
            environment.render()

            reward = float(timestep.reward or 0.0)
            done = float(timestep.last())
            next_obs = timestep.observation

            agent.observe(obs, action, reward, next_obs, done)

            if agent.should_update() and (step % 2 == 0):
                agent.update()

            episode_return += reward

            if timestep.last():
                break

        episode_returns.append(episode_return)
        episode_success.append(1.0 if timestep.step_type == dm_env.StepType.LAST and (timestep.reward or 0.0) >= 20.0 else 0.0)

        if episode % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            success_rate = np.mean(episode_success[-10:]) * 100.0
            print(f"Episode {episode}, Return moyen: {avg_return:.2f}, Succès(10): {success_rate:.1f}%")

    environment.close()
    return agent, episode_returns


def plot_training_returns(returns):
    if plt is None:
        print("matplotlib non disponible: graphique désactivé.")
        return

    x = np.arange(len(returns))
    y = np.asarray(returns, dtype=np.float32)
    window = 10

    if len(y) >= window:
        moving_avg = np.convolve(y, np.ones(window) / window, mode="valid")
        moving_x = np.arange(window - 1, len(y))
    else:
        moving_avg = y
        moving_x = x

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.35, label="Return épisode")
    plt.plot(moving_x, moving_avg, linewidth=2.0, label="Moyenne mobile (10)")
    plt.title("Progression de l'entraînement UR5e")
    plt.xlabel("Épisode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# UTILISATION JAX POUR L'OPTIMISATION
# ============================================================================

@jax.jit
def compute_policy_loss(params, obs_batch):
    """Exemple de calcul JIT avec JAX"""
    # Version optimisée du loss
    return jnp.mean(obs_batch ** 2)  # Placeholder

@jax.jit
def compute_q_loss(params, obs_batch, action_batch, target_q):
    """Loss Q avec JIT"""
    return jnp.mean((obs_batch - target_q) ** 2)  # Placeholder


if __name__ == "__main__":
    print("="*60)
    print("UR5e REACH AVEC JAX + MuJoCo (sans Acme)")
    print("="*60)
    print("Framework: Agent JAX local")
    print("Backend: JAX (compilation XLA)")
    print("Simulateur: MuJoCo")
    print("="*60)
    
    trained_agent, returns = train_ur5e_acme()
    plot_training_returns(returns)
    
    print("\nEntraînement terminé!")
    print(f"Meilleur return: {max(returns):.2f}")
