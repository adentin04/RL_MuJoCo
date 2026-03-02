## Gymnasium 
Step 1 : I started to code with it because it is simpler to understand the RL thing
I start with this :

State: 4 numbers (cart position, velocity, pole angle, pole velocity)

Action: 0 or 1 (push left or right)

Reward: +1 for every timestep the pole stays up

Episode ends when pole falls

This was the code :
``` Python
import gymnasium as gym

env = gym.make("CartPole-v1")

state, info = env.reset()

  

print("state(4 numbers) and info:", state, info)

 [cart_position, cart_velocity, pole_angle, pole_velocity]

  

action = env.action_space.sample() # 0 ou 1 aléatoire

print("action:", action )

  

next_state, reward, terminated, truncated, info = env.step(action)

print("next_state:", next_state)

print("reward:", reward) # +1 si le pole est encore debout

print("terminated:", terminated) # True si le pole est tombé ou si le cart est sorti de la zone

env.close()
```
This is the output :
![[Pasted image 20260302143935.png]]
![[Pasted image 20260302144114.png]]

Step 2 — Implement an agent 

A random policy (choose random action)

Then a basic policy gradient or Q-learning

Even a 100-line implementation is enough.

You need to feel:

How the policy outputs actions

How rewards are stored

How loss is computed

How gradients update parameters

Without this, Acme will feel abstract and confusing.

```Python
import gymnasium as gym

env = gym.make("CartPole-v1")

state, info = env.reset()

total_reward = [] # On initialise une liste pour stocker les récompenses de chaque épisode

for episode in range(10): # On va faire 10 épisodes

    state, info = env.reset() # On réinitialise l'environnement à chaque épisode

    episode_reward = 0 # On initialise une variable pour stocker la récompense totale de l'épisode

        action = env.action_space.sample() # 0 ou 1 aléatoire

        next_state, reward, terminated, truncated, info = env.step(action) # On effectue l'action et on observe le résultat

        episode_reward += reward # On ajoute la récompense de cette étape à la récompense totale de l'épisode

        if terminated or truncated: # Si l'épisode est terminé (pole tombé ou cart sorti de la zone)

            break # On sort de la boucle pour passer à l'épisode suivant

        total_reward.append(episode_reward)

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}") # On affiche la récompense totale de l'épisode

  

print("state(4 numbers) and info:", state, info)

print("action:", action )

  
  

print("next_state:", next_state)

print("reward:", reward) # +1 si le pole est encore debout

print("terminated:", terminated) # True si le pole est tombé ou si le cart est sorti de la zone

env.close()

print(f"Moyenne: {sum(total_reward)/len(total_reward)}") # On affiche la moyenne des récompenses
```

Output : 
this is what a episode look like :![[Pasted image 20260302150614.png]]


![[Pasted image 20260302150504.png]]## Policy Gradient simple (REINFORCE)

``` python 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
# Réseau de politique : state (4) -> action (2)
class PolicyNet(nn.Module): # On définit un réseau de neurones simple avec une couche cachée
    def __init__(self): # On initialise les poids du réseau
        super().__init__() # On appelle le constructeur de la classe parente
        self.fc = nn.Sequential( # On crée une séquence de couches linéaires et d'activations
            nn.Linear(4, 128), # Couche d'entrée : 4 neurones (état)
            nn.ReLU(), # Fonction d'activation ReLU
            nn.Linear(128, 2), # Couche de sortie : 2 neurones (actions)
            nn.Softmax(dim=-1) # Fonction d'activation Softmax pour obtenir des probabilités
        )
    def forward(self, x): #On définit la fonction de passage avant du réseau
        return self.fc(x) # On retourne les probabilités d'action
    
env = gym.make("CartPole-v1") # On crée l'environnement CartPole-v1
policy = PolicyNet() # On instancie le réseau de politique 
optimizer = optim.Adam(policy.parameters(), lr=1e-2) # On utilise l'optimiseur Adam pour entraîner le réseau
gamma = 0.99 # facteur de discount pour les récompenses futures


total_reward = [] # On initialise une liste pour stocker les récompenses de chaque épisode
for episode in range(10): # On va faire 10 épisodes
    state, info = env.reset() # On réinitialise l'environnement à chaque épisode
    episode_reward = 0 # On initialise une variable pour stocker la récompense totale de l'épisode
    log_probs = [] # On initialise une liste pour stocker les log-probabilités
    episode_rewards = [] # Rewards de cet épisode uniquement
    step = 0
    print(f"\n{'='*50}")
    print(f"EPISODE {episode + 1} — état initial: {state.round(3)}")
    # Jouer un épisode
    while True:
        state_t = torch.tensor(state, dtype=torch.float32) # On convertit l'état en tenseur PyTorch
        probs = policy(state_t) # On passe l'état à travers le réseau pour obtenir les probabilités d'action
        dist = torch.distributions.Categorical(probs) # On crée une distribution catégorielle à partir des probabilités d'action
        action = dist.sample() # On échantillonne une action à partir de la distribution
        
        if step < 3:  # On affiche seulement les 3 premiers steps pour ne pas surcharger
            print(f"  Step {step+1}: probs=[gauche:{probs[0].item():.2f}, droite:{probs[1].item():.2f}] => action={'GAUCHE' if action.item()==0 else 'DROITE'}")

        log_probs.append(dist.log_prob(action)) # On stocke le log-probabilité de l'action choisie pour l'entraînement ultérieur
        next_state, reward, terminated, truncated, info = env.step(action.item()) # On effectue l'action et on observe le résultat
        total_reward.append(reward) # On stocke la récompense obtenue pour l'entraînement ultérieur
        episode_rewards.append(reward)
        episode_reward += reward
        step += 1
        state = next_state
      
        if terminated or truncated: # Si l'épisode est terminé (pole tombé ou cart sorti de la zone)
            print(f"  ... ({step} steps au total)")
            print(f"  Episode terminé — reward total: {episode_reward}")
            break # On sort de la boucle pour passer à l'épisode suivant
    returns = [] # On initialise une liste pour stocker les retours (cumulative rewards)
    G = 0 # On initialise la variable de retour à 0
    for r in reversed(episode_rewards): # On calcule les retours en partant de la fin de l'épisode 
        G = r + gamma * G # On met à jour le retour en ajoutant la récompense actuelle et en appliquant le facteur de discount
        returns.insert(0, G) # On insère le retour au début de la liste
    returns = torch.tensor(returns, dtype=torch.float32) # On convertit les retours en tenseur PyTorch
    returns = (returns - returns.mean()) / (returns.std() + 1e-9) # On normalise les retours pour stabiliser l'entraînement
    loss = 0 # On initialise la variable de perte à 0
    for log_p , G_t in zip(log_probs, returns): # On calcule la perte en multipliant les log-probabilités par les retours
        loss += -log_p * G_t # On ajoute la contribution de chaque action à la perte totale
    optimizer.zero_grad() # On réinitialise les gradients de l'optimiseur
    loss.backward() # On effectue la rétropropagation pour calculer les gradients
    optimizer.step() # On met à jour les poids du réseau en fonction des gradients calculés
    print(f"  Loss: {loss.item():.4f} — mise à jour des poids effectuée")

print("state(4 numbers) and info:", state, info)
print("action:", action )


print("next_state:", next_state)
print("reward:", reward) # +1 si le pole est encore debout
print("terminated:", terminated) # True si le pole est tombé ou si le cart est sorti de la zone
env.close()
print(f"Moyenne: {sum(total_reward)/len(total_reward)}") # On affiche la moyenne des récompenses    
``` 
output : 
``` bash
==================================================
EPISODE 1 — état initial: [-0.018  0.017 -0.003 -0.011]
  Step 1: probs=[gauche:0.58, droite:0.42] => action=DROITE
  Step 2: probs=[gauche:0.58, droite:0.42] => action=DROITE
  Step 3: probs=[gauche:0.58, droite:0.42] => action=GAUCHE
  ... (36 steps au total)
  Episode terminé — reward total: 36.0
  Loss: 0.0793 — mise à jour des poids effectuée

==================================================
EPISODE 2 — état initial: [-0.003 -0.045 -0.036 -0.024]
  Step 1: probs=[gauche:0.55, droite:0.45] => action=GAUCHE
  Step 2: probs=[gauche:0.54, droite:0.46] => action=DROITE
  Step 3: probs=[gauche:0.55, droite:0.45] => action=DROITE
  ... (25 steps au total)
  Episode terminé — reward total: 25.0
  Loss: -0.4460 — mise à jour des poids effectuée

==================================================
EPISODE 3 — état initial: [0.044 0.004 0.012 0.002]
  Step 1: probs=[gauche:0.58, droite:0.42] => action=GAUCHE
  Step 2: probs=[gauche:0.55, droite:0.45] => action=GAUCHE       
  Step 3: probs=[gauche:0.54, droite:0.46] => action=GAUCHE       
  ... (12 steps au total)
  Episode terminé — reward total: 12.0
  Loss: -0.3635 — mise à jour des poids effectuée

==================================================
EPISODE 4 — état initial: [-0.02   0.044  0.043  0.013]
  Step 1: probs=[gauche:0.63, droite:0.37] => action=GAUCHE       
  Step 2: probs=[gauche:0.61, droite:0.39] => action=GAUCHE       
  Step 3: probs=[gauche:0.60, droite:0.40] => action=GAUCHE       
  ... (10 steps au total)
  Episode terminé — reward total: 10.0
  Loss: -0.6225 — mise à jour des poids effectuée

==================================================
EPISODE 5 — état initial: [-0.037  0.01  -0.044 -0.027]
  Step 1: probs=[gauche:0.71, droite:0.29] => action=GAUCHE       
  Step 2: probs=[gauche:0.69, droite:0.31] => action=DROITE       
  Step 3: probs=[gauche:0.71, droite:0.29] => action=DROITE       
  ... (18 steps au total)
  Episode terminé — reward total: 18.0
  Loss: 3.4321 — mise à jour des poids effectuée

==================================================
EPISODE 6 — état initial: [-0.024 -0.037  0.012 -0.03 ]
  Step 1: probs=[gauche:0.72, droite:0.28] => action=DROITE       
  Step 2: probs=[gauche:0.75, droite:0.25] => action=GAUCHE
  Step 3: probs=[gauche:0.72, droite:0.28] => action=DROITE       
  ... (22 steps au total)
  Episode terminé — reward total: 22.0
  Loss: -0.9852 — mise à jour des poids effectuée

==================================================
EPISODE 7 — état initial: [-0.001  0.033  0.022  0.03 ]
  Step 1: probs=[gauche:0.72, droite:0.28] => action=GAUCHE       
  Step 2: probs=[gauche:0.70, droite:0.30] => action=DROITE       
  Step 3: probs=[gauche:0.72, droite:0.28] => action=GAUCHE       
  ... (16 steps au total)
  Episode terminé — reward total: 16.0
  Loss: 1.5718 — mise à jour des poids effectuée

==================================================
EPISODE 8 — état initial: [ 0.002 -0.01  -0.    -0.034]
  Step 1: probs=[gauche:0.74, droite:0.26] => action=GAUCHE       
  Step 2: probs=[gauche:0.70, droite:0.30] => action=GAUCHE       
  Step 3: probs=[gauche:0.70, droite:0.30] => action=DROITE       
  ... (14 steps au total)
  Loss: -1.1635 — mise à jour des poids effectuée

==================================================
EPISODE 9 — état initial: [ 0.027  0.03  -0.013 -0.011]
  Step 1: probs=[gauche:0.74, droite:0.26] => action=DROITE
  Step 2: probs=[gauche:0.79, droite:0.21] => action=DROITE
  Step 3: probs=[gauche:0.83, droite:0.17] => action=GAUCHE
  ... (23 steps au total)
  Episode terminé — reward total: 23.0
  Loss: 0.8797 — mise à jour des poids effectuée

==================================================
EPISODE 10 — état initial: [ 0.022 -0.002  0.026  0.049]
  Step 1: probs=[gauche:0.73, droite:0.27] => action=DROITE
  Step 2: probs=[gauche:0.78, droite:0.22] => action=GAUCHE
  Step 3: probs=[gauche:0.73, droite:0.27] => action=GAUCHE
  ... (26 steps au total)
  Episode terminé — reward total: 26.0
  Loss: 1.3161 — mise à jour des poids effectuée
state(4 numbers) and info: [-0.00510058 -0.42723414  0.22283566  1.4391093 ] {}
action: tensor(0)
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
  Loss: -1.1635 — mise à jour des poids effectuée

==================================================
EPISODE 9 — état initial: [ 0.027  0.03  -0.013 -0.011]
  Step 1: probs=[gauche:0.74, droite:0.26] => action=DROITE
  Step 2: probs=[gauche:0.79, droite:0.21] => action=DROITE
  Step 3: probs=[gauche:0.83, droite:0.17] => action=GAUCHE
  ... (23 steps au total)
  Episode terminé — reward total: 23.0
  Loss: 0.8797 — mise à jour des poids effectuée

==================================================
EPISODE 10 — état initial: [ 0.022 -0.002  0.026  0.049]
  Step 1: probs=[gauche:0.73, droite:0.27] => action=DROITE
  Step 2: probs=[gauche:0.78, droite:0.22] => action=GAUCHE
  Step 3: probs=[gauche:0.73, droite:0.27] => action=GAUCHE
  ... (26 steps au total)
  Episode terminé — reward total: 26.0
  Loss: 1.3161 — mise à jour des poids effectuée
state(4 numbers) and info: [-0.00510058 -0.42723414  0.22283566  1.4391093 ] {}
action: tensor(0)
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
  Episode terminé — reward total: 23.0
  Loss: 0.8797 — mise à jour des poids effectuée

==================================================
EPISODE 10 — état initial: [ 0.022 -0.002  0.026  0.049]
  Step 1: probs=[gauche:0.73, droite:0.27] => action=DROITE
  Step 2: probs=[gauche:0.78, droite:0.22] => action=GAUCHE
  Step 3: probs=[gauche:0.73, droite:0.27] => action=GAUCHE
  ... (26 steps au total)
  Episode terminé — reward total: 26.0
  Loss: 1.3161 — mise à jour des poids effectuée
state(4 numbers) and info: [-0.00510058 -0.42723414  0.22283566  1.4391093 ] {}
action: tensor(0)
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
  Step 3: probs=[gauche:0.73, droite:0.27] => action=GAUCHE
  ... (26 steps au total)
  Episode terminé — reward total: 26.0
  Loss: 1.3161 — mise à jour des poids effectuée
state(4 numbers) and info: [-0.00510058 -0.42723414  0.22283566  1.4391093 ] {}
action: tensor(0)
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
state(4 numbers) and info: [-0.00510058 -0.42723414  0.22283566  1.4391093 ] {}
action: tensor(0)
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
next_state: [-0.00510058 -0.42723414  0.22283566  1.4391093 ]
reward: 1.0
reward: 1.0
terminated: True
Moyenne: 1.0
```