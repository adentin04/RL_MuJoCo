import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class SimplePolicy:
    def __init__(self, obs_dim, n_actions, lr=0.01): #self -> this m obs_dim 
        self.weights = np.random.randn(obs_dim, n_actions) * 0.01
        self.lr = lr
        
    def get_action_probs(self, obs):
        # Softmax correct et stable
        logits = obs @ self.weights
        logits = logits - np.max(logits)  # Pour stabilité
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)
    
    def sample_action(self, obs):
        probs = self.get_action_probs(obs)
        return np.random.choice(len(probs), p=probs)
    
    def update(self, trajectory):
        observations = np.array([t[0] for t in trajectory])
        actions = np.array([t[1] for t in trajectory])
        rewards = np.array([t[2] for t in trajectory])
        
        # 1. Calcul des retours (discounted returns)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        # 2. Normalisation (optionnel mais aide)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 3. Calcul du gradient CORRECT pour REINFORCE
        grad = np.zeros_like(self.weights)
        
        for obs, action, G in zip(observations, actions, returns):
            probs = self.get_action_probs(obs)
            
            # Le gradient du log-probabilité pour softmax est:
            # Pour l'action choisie: obs * (1 - prob_action)
            # Pour les autres actions: -obs * prob_autre_action
            
            # Version plus simple: on calcule d'abord le gradient pour toutes les actions
            for a in range(len(probs)):
                if a == action:
                    grad[:, a] += obs * (1 - probs[a]) * G
                else:
                    grad[:, a] -= obs * probs[a] * G
        
        # 4. Mise à jour
        self.weights += self.lr * grad / len(observations)

def train_simple():
    env = gym.make("CartPole-v1")
    policy = SimplePolicy(4, 2, lr=0.1)  # J'ai augmenté le learning rate
    
    reward_history = []
    best_reward = 0
    
    for episode in range(10000):  # Plus d'épisodes
        obs, _ = env.reset()
        trajectory = []
        total_reward = 0
        
        for t in range(10000):
            action = policy.sample_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            trajectory.append((obs, action, reward))
            obs = next_obs
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Mise à jour après l'épisode
        policy.update(trajectory)
        reward_history.append(total_reward)
        
        # Track du meilleur score
        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode:4d}, Reward: {total_reward:3.0f}, "
                  f"Avg (50): {avg_reward:5.1f}, Best: {best_reward:3.0f}")
    
    env.close()
    return reward_history

if __name__ == "__main__":
    rewards = train_simple()
    
    # Visualisation
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Episode reward')
    # Moyenne mobile
    moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(rewards)), moving_avg, 'r-', label='Moving avg (50)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole avec REINFORCE corrigé')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards[-200:], bins=20)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution des 200 derniers épisodes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    