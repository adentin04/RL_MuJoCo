"""UR5e Reach Environment pour Acme/JAX avec MuJoCo - Version Recherche"""

import dm_env
from dm_env import specs
import numpy as np
import jax
import jax.numpy as jnp
from acme import core, types
from acme.jax import networks, utils
from acme.agents.jax import sac
from acme.jax.layouts import distributed_layout
import mujoco
from typing import Optional, Dict, Any, Tuple
import dataclasses

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
        
        # Bonus progression
        z_bonus = 0.3 if ee_pos[2] > self._state.last_distance else 0.0
        
        # Pénalité douceur
        smoothness = -0.001 * float(np.mean(np.abs(self.data.ctrl[:6])))
        
        # Bonus succès
        success = 10.0 if distance < 0.05 else 0.0
        
        return float(main_reward + z_reward + z_bonus + smoothness + success)
    
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
        if self._viewer is None and self._render_mode:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
    
    def close(self):
        if self._viewer:
            self._viewer.close()


# ============================================================================
# CONFIGURATION JAX + ACME
# ============================================================================

def make_networks(environment_spec):
    """Création des réseaux JAX pour SAC"""
    
    # Dimensions
    obs_dim = environment_spec.observations.shape[0]  # 21
    act_dim = environment_spec.actions.shape[0]       # 6
    
    # Policy network (acteur) en JAX
    def policy_network(obs: jnp.ndarray) -> jnp.ndarray:
        # Architecture 256-256
        hidden = jnp.tanh(jnp.dot(obs, jnp.ones((obs_dim, 256))))  # Placeholder
        hidden = jnp.tanh(jnp.dot(hidden, jnp.ones((256, 256))))
        action = jnp.tanh(jnp.dot(hidden, jnp.ones((256, act_dim))))
        return action * 1.0  # Scale to [-1, 1]
    
    # Q-network (critique) en JAX
    def q_network(obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([obs, action])
        hidden = jnp.tanh(jnp.dot(concat, jnp.ones((obs_dim + act_dim, 256))))
        hidden = jnp.tanh(jnp.dot(hidden, jnp.ones((256, 256))))
        q_value = jnp.dot(hidden, jnp.ones((256, 1)))
        return q_value.squeeze()
    
    # Retourner les networks sous format Acme
    return {
        'policy': networks.FeedForwardNetwork(
            init=lambda rng: (),
            apply=lambda params, obs: policy_network(obs)
        ),
        'q': networks.FeedForwardNetwork(
            init=lambda rng: (),
            apply=lambda params, obs, action: q_network(obs, action)
        )
    }


# ============================================================================
# AGENT SAC JAX (DeepMind Acme)
# ============================================================================

def create_sac_agent(environment):
    """Création de l'agent SAC JAX Acme"""
    
    environment_spec = specs.make_environment_spec(environment)
    networks = make_networks(environment_spec)
    
    # Config SAC
    sac_config = sac.SACConfig(
        learning_rate=3e-4,
        discount=0.99,
        tau=0.005,
        init_temperature=1.0,
        target_entropy=-6.0,  # Pour 6 actions
        batch_size=256,
        min_replay_size=10000,
        max_replay_size=1000000,
    )
    
    # Agent SAC JAX
    agent = sac.SAC(
        environment_spec=environment_spec,
        policy_network=networks['policy'],
        q_network=networks['q'],
        config=sac_config,
    )
    
    return agent


# ============================================================================
# BOUCLE D'ENTRAÎNEMENT ACME
# ============================================================================

def train_ur5e_acme():
    """Entraînement avec Acme + JAX"""
    
    # Créer l'environnement
    environment = UR5eReachEnvDM(render_mode='human')
    
    # Créer l'agent
    agent = create_sac_agent(environment)
    
    # Boucle d'entraînement Acme
    num_episodes = 1000
    max_steps_per_episode = 500
    
    # Pour le suivi
    episode_returns = []
    
    for episode in range(num_episodes):
        timestep = environment.reset()
        episode_return = 0.0
        
        for step in range(max_steps_per_episode):
            # Agent choisit une action
            action = agent.select_action(timestep.observation)
            
            # Step environnement
            timestep = environment.step(action)
            
            # Agent observe la transition
            agent.observe(action, timestep)
            
            # Mise à jour de l'agent (si ready)
            if agent.should_update():
                agent.update()
            
            episode_return += timestep.reward
            
            if timestep.last():  # terminated ou truncated
                break
        
        episode_returns.append(episode_return)
        
        if episode % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(f"Episode {episode}, Return moyen: {avg_return:.2f}")
    
    return agent, episode_returns


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
    # Lancement de l'entraînement
    print("="*60)
    print("UR5e REACH AVEC ACME + JAX")
    print("="*60)
    print("Framework: DeepMind Acme")
    print("Backend: JAX (compilation XLA)")
    print("Simulateur: MuJoCo")
    print("="*60)
    
    trained_agent, returns = train_ur5e_acme()
    
    print("\nEntraînement terminé!")
    print(f"Meilleur return: {max(returns):.2f}")