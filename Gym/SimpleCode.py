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
for episode in range(1000): # On va faire 1000 épisodes pour que le réseau ait le temps d'apprendre
    state, info = env.reset() # On réinitialise l'environnement à chaque épisode
    episode_rewards = [] # Rewards de cet épisode uniquement (CORRIGÉ : séparé de total_reward)
    log_probs = [] # On initialise une liste pour stocker les log-probabilités
    # Jouer un épisode
    while True:
        state_t = torch.tensor(state, dtype=torch.float32) # On convertit l'état en tenseur PyTorch
        probs = policy(state_t) # On passe l'état à travers le réseau pour obtenir les probabilités d'action
        dist = torch.distributions.Categorical(probs) # On crée une distribution catégorielle à partir des probabilités d'action
        action = dist.sample() # On échantillonne une action à partir de la distribution

        log_probs.append(dist.log_prob(action)) # On stocke le log-probabilité de l'action choisie pour l'entraînement ultérieur
        next_state, reward, terminated, truncated, info = env.step(action.item()) # On effectue l'action et on observe le résultat
        episode_rewards.append(reward) # On stocke la récompense de cet épisode uniquement (CORRIGÉ)

        state = next_state # On met à jour l'état courant
        if terminated or truncated: # Si l'épisode est terminé (pole tombé ou cart sorti de la zone)
            break # On sort de la boucle pour passer à l'épisode suivant

    # Calcul des retours sur l'épisode uniquement
    returns = [] # On initialise une liste pour stocker les retours (cumulative rewards)
    G = 0 # On initialise la variable de retour à 0
    for r in reversed(episode_rewards): # On calcule les retours en partant de la fin de l'épisode
        G = r + gamma * G # On met à jour le retour en ajoutant la récompense actuelle et en appliquant le facteur de discount
        returns.insert(0, G) # On insère le retour au début de la liste

    returns = torch.tensor(returns, dtype=torch.float32) # On convertit les retours en tenseur PyTorch
    returns = (returns - returns.mean()) / (returns.std() + 1e-9) # On normalise les retours pour stabiliser l'entraînement

    loss = 0 # On initialise la variable de perte à 0
    for log_p, G_t in zip(log_probs, returns): # On calcule la perte en multipliant les log-probabilités par les retours
        loss += -log_p * G_t # On ajoute la contribution de chaque action à la perte totale

    optimizer.zero_grad() # On réinitialise les gradients de l'optimiseur
    loss.backward() # On effectue la rétropropagation pour calculer les gradients
    optimizer.step() # On met à jour les poids du réseau en fonction des gradients calculés

    total_reward.append(sum(episode_rewards)) # On stocke la récompense totale de cet épisode
    if episode % 100 == 0: # On affiche la progression tous les 100 épisodes
        print(f"Episode {episode}, Reward: {sum(episode_rewards)}")

print("state(4 numbers) and info:", state, info)
print("action:", action )


print("next_state:", next_state)
print("reward:", reward) # +1 si le pole est encore debout
print("terminated:", terminated) # True si le pole est tombé ou si le cart est sorti de la zone
env.close()
print(f"Moyenne: {sum(total_reward)/len(total_reward)}") # On affiche la moyenne des récompenses    