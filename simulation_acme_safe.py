"""UR5e Reach Environment - version Acme-safe (sans imports Acme conflictuels)."""

import dataclasses
from typing import Optional, Dict, Any, Tuple

import dm_env
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from dm_env import specs

try:
    import mujoco.viewer as mj_viewer
except Exception:
    mj_viewer = None

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
        self._viewer_warning_printed = False
        
    def _initialize_state(self) -> UR5eState:
        """Crée l'état initial"""
        return UR5eState(
            qpos=jnp.array(self.data.qpos[:6].copy()),
            qvel=jnp.array(self.data.qvel[:6].copy()),
            target=self.curriculum_targets[0],
            time=0.0,
            last_distance=np.inf,
            curriculum_stage=0
        )
    
    def reset(self) -> dm_env.TimeStep:
        """Reset dm_env style"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale avec bruit
        initial_qpos = self.home_joints + np.random.uniform(-0.1, 0.1, 6)
        self.data.qpos[:6] = initial_qpos
        self.data.ctrl[:6] = initial_qpos
        
        # Mettre à jour la cible (sphère rouge)
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_marker')
        if target_body_id >= 0:
            mocap_id = self.model.body_mocapid[target_body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = self.curriculum_targets[0]
        
        mujoco.mj_forward(self.model, self.data)
        
        # Mettre à jour l'état JAX
        self._state = UR5eState(
            qpos=jnp.array(initial_qpos),
            qvel=jnp.zeros(6),
            target=self.curriculum_targets[0],
            time=0.0,
            last_distance=np.inf,
            curriculum_stage=0
        )
        
        # Premier timestep dm_env
        return dm_env.restart(self._get_obs())
    
    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Step dm_env style"""
        # Appliquer l'action
        self.data.ctrl[:6] = self.home_joints + action * self.action_scale
        
        # Simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
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
        truncated = new_time > 10.0
        
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
    
    def _get_end_effector_pos_jax(self) -> jnp.ndarray:
        """Position effecteur en JAX"""
        if self.ee_site_id is not None:
            return jnp.array(self.data.site_xpos[self.ee_site_id])
        return jnp.array(self.data.geom_xpos[-1])
    
    def _compute_reward(self) -> float:
        """Calcul récompense"""
        ee_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(ee_pos - np.array(self._state.target))
        
        # Récompense principale
        main_reward = -distance
        
        # Bonus Z
        z_reward = -2.0 * abs(ee_pos[2] - float(self._state.target[2]))
        
        # Bonus progression distance
        progress_bonus = 0.5 if distance < self._state.last_distance else 0.0
        
        # Pénalité douceur
        smoothness = -0.001 * float(np.mean(np.abs(self.data.ctrl[:6])))
        
        # Bonus succès
        success = 10.0 if distance < 0.05 else 0.0
        
        return float(main_reward + z_reward + progress_bonus + smoothness + success)
    
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
        """Rendu MuJoCo"""
        if mj_viewer is None:
            if not self._viewer_warning_printed:
                print("Viewer MuJoCo indisponible dans cette installation (mujoco.viewer).")
                self._viewer_warning_printed = True
            return

        if self._viewer is None and self._render_mode:
            self._viewer = mj_viewer.launch_passive(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
    
    def close(self):
        if self._viewer:
            self._viewer.close()


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
        self.obs[i] = transition.obs
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.next_obs[i] = transition.next_obs
        self.dones[i] = transition.done
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
        obs_j = jnp.asarray(obs)
        action = np.asarray(actor_forward(self.actor_params, obs_j), dtype=np.float32)
        if explore:
            action = action + self.np_rng.normal(0.0, self.noise_std, size=action.shape).astype(np.float32)
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

        def critic_loss_fn(c_params):
            q = jax.vmap(lambda o, a: critic_forward(c_params, o, a))(obs, actions)
            return jnp.mean((q - y) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_params = tree_sgd_step(critic_params, critic_grads, critic_lr)

        def actor_loss_fn(a_params):
            pred_actions = jax.vmap(lambda o: actor_forward(a_params, o))(obs)
            q_pred = jax.vmap(lambda o, a: critic_forward(critic_params, o, a))(obs, pred_actions)
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


def train_ur5e_acme_safe(
    render: bool = False,
    num_episodes: int = 300,
    max_steps_per_episode: int = 500,
    step_heartbeat: int = 100,
):
    """Entraînement style Acme avec agent JAX local (sans dépendances Acme runtime)."""

    environment = UR5eReachEnvDM(render_mode='human' if render else None)

    obs_dim = int(environment.observation_spec().shape[0])
    act_dim = int(environment.action_spec().shape[0])
    agent = JAXReachAgent(obs_dim=obs_dim, action_dim=act_dim)

    episode_returns = []

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

            if step_heartbeat > 0 and (step + 1) % step_heartbeat == 0:
                print(f"Episode {episode + 1}/{num_episodes} - step {step + 1}/{max_steps_per_episode}")

            if timestep.last():
                break

        episode_returns.append(episode_return)

        avg_window = 10 if len(episode_returns) >= 10 else len(episode_returns)
        avg_return = np.mean(episode_returns[-avg_window:])
        print(f"Episode {episode + 1}/{num_episodes}, Return: {episode_return:.2f}, Moyenne({avg_window}): {avg_return:.2f}")

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
    print("UR5e REACH - VERSION ACME-SAFE")
    print("="*60)
    print("Framework: Agent JAX local")
    print("Backend: JAX (compilation XLA)")
    print("Simulateur: MuJoCo")
    print("="*60)
    
    trained_agent, returns = train_ur5e_acme_safe(render=False)
    plot_training_returns(returns)
    
    print("\nEntraînement terminé!")
    print(f"Meilleur return: {max(returns):.2f}")
