import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite

def obs_to_vec(timestep):
    return np.concatenate([v.ravel() for v in timestep.observation.values()])

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        std = torch.exp(self.logstd)
        return mu, std

def run_episode(env, policy, device):
    obs = env.reset()
    obs_v = obs_to_vec(obs)
    log_probs = []
    rewards = []
    done = False
    while not obs.last():
        x = torch.tensor(obs_v, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std = policy(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample().squeeze(0).cpu().numpy()
        # clip to action spec
        spec = env.action_spec()
        action = np.clip(action, spec.minimum, spec.maximum)
        timestep = env.step(action)
        r = 0.0 if timestep.reward is None else float(timestep.reward)
        rewards.append(r)
        log_probs.append(dist.log_prob(torch.tensor(action, device=device)).sum())
        obs_v = obs_to_vec(timestep)
        obs = timestep
    return log_probs, rewards

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train(n_episodes=200, lr=1e-3, device='cpu'):
    env = suite.load('cartpole', 'balance')
    # get obs dim
    obs = env.reset()
    obs_dim = obs_to_vec(obs).shape[0]
    act_dim = env.action_spec().shape[0]
    policy = PolicyNet(obs_dim, act_dim).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(n_episodes):
        logps, rewards = run_episode(env, policy, device)
        if len(rewards) == 0:
            continue
        returns = compute_returns(rewards)
        loss = 0.0
        for lp, G in zip(logps, returns):
            loss = loss - lp * G
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Ep {ep+1}\tReturn {sum(rewards):.2f}")

if __name__ == '__main__':
    train()