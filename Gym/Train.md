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
<<<<<<< HEAD

Il y a eu  des changements de code , on a remarque que JAX n'utilisez pas la GPU donc on a change , j'ai installe cuda , les drivers de nvidia et j'ai modifie le code, maintenant on a aussi un rendu graphique pour voir le cartpole:
``` python
import gymnasium as gym

import matplotlib.pyplot as plt  # Pour tracer la courbe d'apprentissage

import matplotlib.patches as mpatches  # Pour les légendes du graphique

import time  # Pour les délais entre les frames (ralentir l'animation)

  

import jax  # On importe jax pour les calculs différentiables et les opérations sur GPU/TPU

import jax.numpy as jnp # On importe jax.numpy pour les opérations sur les tableaux, similaire à numpy mais optimisé pour jax

  

# =============================================================================

# STRATÉGIE UTILISÉE : REINFORCE (Monte-Carlo Policy Gradient)

# -----------------------------------------------------------------------------

# L'agent apprend une POLITIQUE STOCHASTIQUE (réseau linéaire) qui mappe

# directement les observations vers des probabilités d'action.

# À chaque épisode :

#   1. L'agent joue un épisode COMPLET avec la politique actuelle

#   2. On calcule les retours cumulés actualisés (Monte-Carlo)

#   3. On fait une descente de gradient pour maximiser les actions qui

#      ont mené à de bonnes récompenses (gradient de politique)

# Pas de valeur d'état (pas de critique), pas de table Q → c'est du

# REINFORCE pur, l'algorithme fondateur des méthodes policy gradient.

# =============================================================================

  

# =============================================================================

# POLITIQUE (POLICY) — règles que l'agent utilise pour choisir ses actions

# -----------------------------------------------------------------------------

# La politique est représentée par un réseau de neurones linéaire à une couche.

# Elle prend en entrée l'état courant (observation) et produit des scores

# (logits) pour chaque action possible. C'est le "cerveau" de l'agent.

# L'agent n'est pas un objet séparé : il est l'ensemble {params + policy_network}.

# =============================================================================

def policy_network(params, observation):

    #Une simple couche linéaire: observation -> logits (scores pour chaque action)

    #params: dictionnaire {'w': poids, 'b':biais}

    # POLITIQUE : transformation linéaire observation → scores d'action

    logits = jnp.dot(observation, params['w']) + params['b'] # calcule des logits en multipliant l'observation par les poids et en ajoutant le biais

    return logits # On retourne les logits, on appliquera softmax plus tard pour obtenir les probabilités d'action

  

# =============================================================================

# CRÉATION DE L'AGENT

# -----------------------------------------------------------------------------

# L'agent est défini par ses paramètres (poids 'w' et biais 'b').

# init_params() crée ces paramètres aléatoires → c'est la naissance de l'agent.

# Plus bas dans le code : params = init_params(rng)  ← instanciation réelle.

# =============================================================================

def init_params(rng, input_dim=4, output_dim=2): # On initialise les paramètres du réseau de politique

    #On crée des paramètres aléatoires pour commencer

    k1, k2 = jax.random.split(rng) # On divise le générateur de nombres aléatoires pour obtenir deux clés

    w = jax.random.normal(k1, (input_dim, output_dim))* 0.1 # On initialise les poids avec une distribution normale et on les scale pour éviter des valeurs trop grandes

    b = jax.random.normal(k2, (output_dim,))*0.1

    params = {'w': w, 'b': b}

    print("\n" + "="*60)

    print("[AGENT] Naissance de l'agent !")

    print(f"  Entrées  : {input_dim} valeurs d'état (position, vitesse, angle, vitesse_angulaire)")

    print(f"  Sorties  : {output_dim} actions possibles (0=gauche, 1=droite)")

    print(f"  Poids W  : matrice {input_dim}x{output_dim}, initialisés aléatoirement")

    print(f"  Biais b  : vecteur de {output_dim}, initialisés aléatoirement")

    print("  → L'agent ne sait RIEN pour l'instant, il agit au hasard.")

    print("="*60)

    return params # On retourne les paramètres sous forme de dictionnaire

  

# =============================================================================

# ACTION — décision prise par l'agent à chaque pas de temps

# -----------------------------------------------------------------------------

# sample_action() applique la politique : observation → action choisie.

# L'action est stochastique (tirée aléatoirement selon les probabilités),

# ce qui permet l'exploration. Dans CartPole, action ∈ {0 (gauche), 1 (droite)}.

# =============================================================================

def sample_action(rng, params, observation, verbose=False): # On définit une fonction pour échantillonner une action à partir de la politique

    logits = policy_network(params, observation) # On calcule les logits à partir de l'observation et des paramètres

    # ACTION : tirage stochastique — la politique est probabiliste, pas déterministe

    action = jax.random.categorical(rng, logits) # On échantillonne une action à partir de la distribution catégorielle définie par les logits

    if verbose:

        probs = jax.nn.softmax(logits)  # convertir logits → probabilités lisibles

        print(f"    [POLITIQUE] Logits bruts        : gauche={float(logits[0]):.3f}, droite={float(logits[1]):.3f}")

        print(f"    [POLITIQUE] Probabilités        : gauche={float(probs[0]):.1%}, droite={float(probs[1]):.1%}")

        print(f"    [ACTION]    Action choisie      : {'GAUCHE (0)' if int(action)==0 else 'DROITE (1)'}")

    return action # On retourne l'action échantillonnée

  

# =============================================================================

# BOUCLE EPISODE — interaction agent ↔ environnement

# =============================================================================

def run_episode(env, params, rng, verbose=False, slow=False): #On définit une fonction pour simuler un épisode dans l'environnement

  

    # STATE (ÉTAT) : vecteur de 4 valeurs [position_chariot, vitesse_chariot,

    # angle_perche, vitesse_angulaire] → représente la situation complète du système

    obs, _ = env.reset() # On réinitialise l'environnement et on obtient l'observation initiale

    done = False # On initialise la variable "done" pour suivre si l'épisode est terminé

    trajectories = [] # Liste pour stocker (obs, action, reward)

    step = 0  # compteur de pas

  

    if verbose:

        print("\n[ENV] Environnement réinitialisé.")

        print(f"  État initial → pos_chariot={obs[0]:.3f}, vit_chariot={obs[1]:.3f}, "

              f"angle_perche={obs[2]:.3f} rad, vit_angulaire={obs[3]:.3f}")

        print("  (Les valeurs sont proches de 0 : perche presque verticale, chariot au centre)")

        print("-"*60)

  

    while not done: # Tant que l'épisode n'est pas terminé

        rng, action_rng = jax.random.split(rng)

        step += 1

        log_this_step = verbose and step <= 4  # on log seulement les 4 premiers pas

  

        if log_this_step:

            print(f"\n  [PAS {step}]")

            print(f"    [ÉTAT]      pos={obs[0]:.3f}, vit={obs[1]:.3f}, "

                  f"angle={obs[2]:.3f} rad ({obs[2]*57.3:.1f}°), vit_ang={obs[3]:.3f}")

  

        # ACTION : l'agent consulte sa politique pour décider quoi faire

        action = sample_action(action_rng, params, obs, verbose=log_this_step)

  

        # appliquer l'action dans l'environnement

        # REWARD (RÉCOMPENSE) : +1 à chaque pas où la perche reste debout

        # next_obs = nouvel ÉTAT après l'action

        next_obs, reward, terminated, truncated, _ = env.step(int(action))

        done = terminated or truncated

  

        if log_this_step:

            print(f"    [REWARD]    Récompense reçue    : +{reward:.1f}  "

                  f"({'épisode TERMINÉ !' if done else 'perche encore debout ✓'})")

        if verbose and step == 4 and not done:

            print("    ... (logs masqués pour les pas suivants) ...")

  

        if slow:

            time.sleep(0.04)  # ~25 fps quand slow=True

        trajectories.append((obs, action, reward))

        obs = next_obs

  

    if verbose:

        print(f"\n[ENV] Épisode terminé après {step} pas.")

        print(f"  Récompense totale : {sum(r for _,_,r in trajectories):.1f}")

        print(f"  (Score max possible dans CartPole-v1 : 500)")

  

    return trajectories # On retourne la liste des trajectoires à la fin de l'épisode

  

# =============================================================================

# MISE À JOUR DE LA POLITIQUE — cœur de l'algorithme REINFORCE

# -----------------------------------------------------------------------------

# compute_loss() calcule la perte du gradient de politique :

#   loss = -Σ log π(a|s) * G   (G = retour cumulé actualisé)

# On minimise cette perte → l'agent augmente la probabilité des actions

# qui ont mené à des retours élevés (apprentissage par récompense différée).

# =============================================================================

def compute_loss(params, trajectories): # On définit une fonction pour calculer la perte à partir des paramètres et des trajectoires

    total_loss = 0.0 # On initialise la perte totale à zéro

    #Calculer les retours (discounted returns)

    returns = [] # Liste pour stocker les retours

    G = 0 # Variable pour calculer le retour cumulé

    gamma = 0.99 # Facteur d'actualisation

    # Parcourir à l'envers pour calculer les retours

    for _,_, reward in reversed(trajectories): # On parcourt les trajectoires à l'envers pour calculer les retours

        G = reward + gamma * G  # On met à jour le retour cumulé en ajoutant la récompense actuelle et en actualisant le retour précédent avec le facteur gamma

        returns.insert(0, G) # On insère le retour calculé au début de la liste des retours

    # Pour chaque étape

    for (obs, action, _), G in zip(trajectories, returns):

        # Calculer log prob de l'action choisie

        logits = policy_network(params, obs) # On calcule les logits à partir de l'observation et des paramètres

        log_probs = jax.nn.log_softmax(logits) # On applique la fonction log_softmax pour obtenir les log-probabilités d'action

        log_prob_action = log_probs[action] # On extrait la log-probabilité de l'action choisie

  

        #loss: -log_prob* return (négatif car on fait de la descente de gradient)

        total_loss += -log_prob_action * G  # ✅ Corrigé

  

    return total_loss / len(trajectories) # On retourne la perte moyenne en divisant la perte totale par le nombre de trajectoires

  

# Jax: gradient et mise à jour

#jax.grad transforme compute_loss en fonction qui retourne les gradients !

grad_loss = jax.grad(compute_loss) # On utilise jax.grad pour obtenir une fonction qui calcule les gradients de la perte par rapport aux paramètres

  

def update_params(params, trajectories, lr=0.01, verbose=False):

    grads = grad_loss(params, trajectories)

    new_params = {

        'w': params['w'] - lr * grads['w'],

        'b': params['b'] - lr * grads['b']

    }

    if verbose:

        grad_norm_w = float(jnp.linalg.norm(grads['w']))

        grad_norm_b = float(jnp.linalg.norm(grads['b']))

        print(f"\n[MISE À JOUR POLITIQUE]")

        print(f"  Norme du gradient W : {grad_norm_w:.5f}")

        print(f"  Norme du gradient b : {grad_norm_b:.5f}")

        print(f"  Taux d'apprentissage : {lr}")

        print(f"  → Les poids sont ajustés pour favoriser les actions rentables.")

    return new_params

# INITIALISATION

# =============================================================================

rng = jax.random.PRNGKey(42) # On initialise une clé de générateur de nombres aléatoires pour jax

  

# CRÉATION DE L'ENVIRONNEMENT : CartPole-v1

# L'environnement simule une perche posée sur un chariot mobile.

# Objectif : maintenir la perche en équilibre le plus longtemps possible.

# La fenetre reste ouverte pendant tout l'entrainement

env = gym.make('CartPole-v1', render_mode='human')  # fenetre visuelle active des le debut

  

# CRÉATION DE L'AGENT : initialisation aléatoire des paramètres (poids + biais)

# C'est ici que l'agent "naît" — il ne sait encore rien, ses poids sont aléatoires.

params = init_params(rng) # On initialise les paramètres du réseau de politique

  

# ÉPISODES POUR LESQUELS ON AFFICHE LES DÉTAILS PAS-À-PAS

VERBOSE_EPISODES = {0, 100, 499}       # épisodes avec logs détaillés pas-à-pas

  

# Liste pour stocker les récompenses de chaque épisode → sert à tracer le graphique

all_rewards = []

  

n_episodes = 500

print("\n" + "="*60)

print("[DÉBUT ENTRAÎNEMENT]")

print(f"  Nombre d'épisodes : {n_episodes}")

print(f"  Algorithme        : REINFORCE (Monte-Carlo Policy Gradient)")

print(f"  Récompense        : +1 par pas de temps où la perche est debout")

print(f"  Objectif          : atteindre 500 pas (score max de CartPole-v1)")

print("="*60)

  

for episode in range(n_episodes):

    rng, episode_rng = jax.random.split(rng)

    verbose = episode in VERBOSE_EPISODES

  

    if verbose:

        print(f"\n{'='*60}")

        print(f"[ÉPISODE {episode}] — logs détaillés activés")

        print(f"{'='*60}")

  

    # 1. Jouer un épisode avec la politique actuelle

    # Ralentir toutes les 10 episodes pour bien voir l'agent evoluer

    slow_render = (episode % 10 == 0)

    trajectories = run_episode(env, params, episode_rng, verbose=verbose, slow=slow_render)

    total_reward = sum([r for _, _, r in trajectories])

    n_steps = len(trajectories)

  

    # 2. Mettre à jour les paramètres

    params = update_params(params, trajectories, lr=0.01, verbose=verbose)

  

    # Enregistrer la récompense pour le graphique

    all_rewards.append(total_reward)

    if episode % 50 == 0 or verbose:

        print(f"\n» Épisode {episode:>3} | Pas : {n_steps:>4} | Récompense totale : {total_reward:>6.1f} "

              f"{'🎯 PARFAIT!' if total_reward >= 500 else ''}")

  

print("\n" + "="*60)

print("[FIN ENTRAÎNEMENT]")

print(f"  Dernier score : {total_reward:.1f} / 500")

print("="*60)

env.close()

  

# =============================================================================

# GRAPHIQUE — Courbe d'apprentissage (récompenses au fil des épisodes)

# =============================================================================

print("\n[GRAPHIQUE] Affichage de la courbe d'apprentissage...")

  

# Calcul d'une moyenne mobile sur 20 épisodes pour lisser la courbe

window = 20

smoothed = [

    sum(all_rewards[max(0, i - window):i + 1]) / len(all_rewards[max(0, i - window):i + 1])

    for i in range(len(all_rewards))

]

  

fig, ax = plt.subplots(figsize=(12, 5))

  

# Courbe brute (transparente)

ax.plot(all_rewards, color='steelblue', alpha=0.3, linewidth=0.8, label='Récompense par épisode')

# Courbe lissée (moyenne mobile)

ax.plot(smoothed, color='steelblue', linewidth=2.0, label=f'Moyenne mobile ({window} épisodes)')

# Ligne du score parfait

ax.axhline(y=500, color='green', linestyle='--', linewidth=1.2, label='Score parfait (500)')

  

ax.set_xlabel('Épisode', fontsize=12)

ax.set_ylabel('Récompense totale (= nombre de pas)', fontsize=12)

ax.set_title('Courbe d\'apprentissage — REINFORCE sur CartPole-v1', fontsize=14, fontweight='bold')

ax.legend(fontsize=10)

ax.set_ylim(0, 520)

ax.grid(True, alpha=0.3)

  

plt.tight_layout()

plt.savefig('learning_curve.png', dpi=120)  # Sauvegarde automatique

plt.show(block=False)  # Affiche sans bloquer la suite du programme

print("[GRAPHIQUE] Courbe sauvegard\u00e9e dans 'learning_curve.png' (fen\u00eatre ouverte en arri\u00e8re-plan)")

  

plt.show()

print("[TERMINE] Ferme la fenetre du graphique pour quitter.")
```
J'ai cree aussi une video pour voir comment le cartPole se comporte: 
Mon idee est aussi de analiser le contenu de la video pour voir si les logs dans le terminal , la video et la courbe d'aprentissage soient coherent . 
Je veux aussi sauvegarder plusieurs log et courbe d'aprentissage dans un dossier pour les analiser et voir la difference . 

Output : 

=======
![[Pasted image 20260304144222.png]]
>>>>>>> aa9505c (ajout de obsidian)
