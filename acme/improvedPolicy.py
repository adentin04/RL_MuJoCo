import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 
from cartpole import SimplePolicy

class ImprovedPolicy(SimplePolicy):
    def __init__(self, obs_dim, n_actions, lr=0.1): 
        super().__init__(obs_dim, n_actions, lr)
        self.baseline = 0  # Pour réduire la variance
        print(f"[ImprovedPolicy] Init -> obs_dim={obs_dim}, n_actions={n_actions}, lr={lr}")
        print(f"[ImprovedPolicy] Weights shape -> {self.weights.shape}")
        
    def update(self, trajectory):
        print("\n[ImprovedPolicy] Update start")
        print(f"[ImprovedPolicy] Trajectory length -> {len(trajectory)}")

        if len(trajectory) == 0:
            print("[ImprovedPolicy] Trajectory vide, update ignoré")
            return

        observations = np.array([t[0] for t in trajectory])
        actions = np.array([t[1] for t in trajectory])
        rewards = np.array([t[2] for t in trajectory])
        print(f"[ImprovedPolicy] Observations shape -> {observations.shape}")
        print(f"[ImprovedPolicy] Actions shape -> {actions.shape}")
        print(f"[ImprovedPolicy] Rewards stats -> min={rewards.min():.3f}, max={rewards.max():.3f}, mean={rewards.mean():.3f}")
        
        # Calcul des returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)
        print(f"[ImprovedPolicy] Returns stats -> min={returns.min():.3f}, max={returns.max():.3f}, mean={returns.mean():.3f}")
        
        # ASTUCE 1: Utiliser une baseline (réduit la variance)
        # Au lieu de returns, on utilise returns - baseline
        # Ça centre les valeurs autour de 0
        old_baseline = self.baseline
        self.baseline = 0.9 * self.baseline + 0.1 * returns.mean()
        advantages = returns - self.baseline
        print(f"[ImprovedPolicy] Baseline -> old={old_baseline:.3f}, new={self.baseline:.3f}")
        
        # ASTUCE 2: Normalisation plus aggressive
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"[ImprovedPolicy] Advantages stats -> min={advantages.min():.3f}, max={advantages.max():.3f}, mean={advantages.mean():.3f}, std={advantages.std():.3f}")
        
        # ASTUCE 3: Gradient clipping (évite les explosions)
        grad = np.zeros_like(self.weights)
        for obs, action, adv in zip(observations, actions, advantages):
            probs = self.get_action_probs(obs)
            for a in range(len(probs)):
                if a == action:
                    grad[:, a] += obs * (1 - probs[a]) * adv
                else:
                    grad[:, a] -= obs * probs[a] * adv
        
        # Clip le gradient
        grad_norm = np.linalg.norm(grad)
        print(f"[ImprovedPolicy] Gradient norm (before clip) -> {grad_norm:.6f}")
        if grad_norm > 1.0:
            grad = grad / grad_norm
            print("[ImprovedPolicy] Gradient clipping applied")
        else:
            print("[ImprovedPolicy] Gradient clipping not needed")
        
        self.weights += self.lr * grad / len(observations)
        print(f"[ImprovedPolicy] Weights updated -> lr={self.lr}, batch_size={len(observations)}")
        print("[ImprovedPolicy] Update done")


def train_improved(episodes=120000, lr=0.5):
    env = gym.make("CartPole-v1")
    policy = ImprovedPolicy(obs_dim=4, n_actions=2, lr=lr)

    reward_history = []
    best_reward = 0.0

    for episode in range(episodes):
        obs, _ = env.reset()
        trajectory = []
        total_reward = 0.0

        for _ in range(10000):
            action = policy.sample_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            trajectory.append((obs, action, reward))
            obs = next_obs
            total_reward += reward

            if terminated or truncated:
                break

        policy.update(trajectory)
        reward_history.append(total_reward)
        best_reward = max(best_reward, total_reward)

        if episode % 25 == 0:
            avg_25 = np.mean(reward_history[-25:])
            print(
                f"[Train] Episode {episode:4d} | reward={total_reward:6.1f} "
                f"| avg25={avg_25:6.1f} | best={best_reward:6.1f}"
            )

    env.close()
    return reward_history


if __name__ == "__main__":
    rewards = train_improved(episodes=120000, lr=0.5)

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.35, label="Reward par épisode")
    moving = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    plt.plot(range(19, len(rewards)), moving, label="Moyenne mobile (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("ImprovedPolicy - CartPole (120000 rewards)")
    plt.grid(True, alpha=0.10)
    plt.legend()
    plt.tight_layout()
    plt.show()