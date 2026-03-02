import gymnasium as gym

import jax  # On importe jax pour les calculs différentiables et les opérations sur GPU/TPU
import jax.numpy as jnp # On importe jax.numpy pour les opérations sur les tableaux, similaire à numpy mais optimisé pour jax

# Policy network 
# en jax , c'est juste une fonction qui prend des paramètres et une entrée
def policy_network(params, observation):
    #Une simple couche linéaire: observation -> logits (scores pour chaque action)
    #params: dictionnaire {'w': poids, 'b':biais}
    logits = jnp.dot(observation, params['w']) + params['b'] # calcule des logits en multipliant l'observation par les poids et en ajoutant le biais
    return logits # On retourne les logits, on appliquera softmax plus tard pour obtenir les probabilités d'action

#Initialiser les paramètres (c'est notre "modèle" au début)
def init_params(rng, input_dim=4, output_dim=2): # On initialise les paramètres du réseau de politique 
    #On crée des paramètres aléatoires pour commencer
    k1, k2 = jax.random.split(rng) # On divise le générateur de nombres aléatoires pour obtenir deux clés
    w = jax.random.normal(k1, (input_dim, output_dim))* 0.1 # On initialise les poids avec une distribution normale et on les scale pour éviter des valeurs trop grandes
    b = jax.random.normal(k2, (output_dim,))*0.1
    return {'w': w, 'b': b} # On retourne les paramètres sous forme de dictionnaire

#La fonction qui choisit une action (notre "policy" en action)
def sample_action(rng, params, observation): # On définit une fonction pour échantillonner une action à partir de la politique
    logits = policy_network(params, observation) # On calcule les logits à partir de l'observation et des paramètres
    action = jax.random.categorical(rng, logits) # On échantillonne une action à partir de la distribution catégorielle définie par les logits
    return action # On retourne l'action échantillonnée

# Simulation d'un épisode

def run_episode(env, params, rng): #On définit une fonction pour simuler un épisode dans l'environnement

    obs, _ = env.reset() # On réinitialise l'environnement et on obtient l'observation initiale
    done = False # On initialise la variable "done" pour suivre si l'épisode est terminé
    trajectories = [] # Liste pour stocker (obs, action, reward)

    while not done: # Tant que l'épisode n'est pas terminé
        # à chaque pas , choisir une action
        rng, action_rng = jax.random.split(rng) # On divise le générateur de nombres aléatoires pour obtenir une nouvelle clé pour l'action
        action = sample_action(action_rng, params, obs) # On échantillonne une action à partir de la politique
        
        # appliquer l'action dans l'environnement
        next_obs, reward, terminated, truncated, _ = env.step(int(action)) # On applique l'action dans l'environnement et on obtient la nouvelle observation, la récompense et les indicateurs de fin d'épisode
        done = terminated or truncated # On met à jour la variable "done" en fonction des indicateurs de fin d'épisode

        #Sauvegarder 
        trajectories.append((obs, action, reward)) # On stocke l'observation, l'action et la récompense dans la liste des trajectoires
        obs = next_obs # On met à jour l'observation pour le prochain pas

    return trajectories # On retourne la liste des trajectoires à la fin de l'épisode

# La fonction de perte (loss) pour reinforce
def compute_loss(params, trajectories): # On définit une fonction pour calculer la perte à partir des paramètres et des trajectoires
    total_loss = 0.0 # On initialise la perte totale à zéro
    #Calculer les retours (discounted returns)
    returns = [] # Liste pour stocker les retours
    G = 0 # Variable pour calculer le retour cumulé
    gamma = 0.99 # Facteur d'actualisation 
    # Parcourir à l'envers pour calculer les retours
    for _,_, reward in reversed(trajectories): # On parcourt les trajectoires à l'envers pour calculer les retours
        G = reward + gamma * G  # On met à jour le retour cumulé en ajoutant la récompense actuelle et en actualisant le retour précédent avec le facteur gamma
        returns.insert(0, G) # On insère le retour calculé au début de la liste des retours
    # Pour chaque étape 
    for (obs, action, _), G in zip(trajectories, returns):
        # Calculer log prob de l'action choisie
        logits = policy_network(params, obs) # On calcule les logits à partir de l'observation et des paramètres
        log_probs = jax.nn.log_softmax(logits) # On applique la fonction log_softmax pour obtenir les log-probabilités d'action
        log_prob_action = log_probs[action] # On extrait la log-probabilité de l'action choisie

        #loss: -log_prob* return (négatif car on fait de la descente de gradient)
        total_loss += -log_prob_action * G  # ✅ Corrigé

    return total_loss / len(trajectories) # On retourne la perte moyenne en divisant la perte totale par le nombre de trajectoires

# Jax: gradient et mise à jour
#jax.grad transforme compute_loss en fonction qui retourne les gradients !
grad_loss = jax.grad(compute_loss) # On utilise jax.grad pour obtenir une fonction qui calcule les gradients de la perte par rapport aux paramètres

def update_params(params, trajectories, lr=0.01): # On définit une fonction pour mettre à jour les paramètres en utilisant les gradients calculés à partir des trajectoires
    grads = grad_loss(params, trajectories) # On calcule les gradients de la perte par rapport aux paramètres
    new_params = { # On crée un nouveau dictionnaire de paramètres mis à jour
        'w': params['w'] - lr * grads['w'], # On met à jour les poids en soustrayant le produit du taux d'apprentissage et des gradients
        'b': params['b'] - lr * grads['b']  # On met à jour les biais de la même manière
    }
    return new_params # On retourne les nouveaux paramètres mis à jour
 # initialisation 
rng = jax.random.PRNGKey(42) # On initialise une clé de générateur de nombres aléatoires pour jax
env = gym.make('CartPole-v1') # On crée l'environnement CartPole-v1 de gym
params = init_params(rng) # On initialise les paramètres du réseau de politique

n_episodes = 500  # On définit le nombre d'épisodes à simuler (plus d'épisodes pour voir l'apprentissage)
for episode in range(n_episodes): # On boucle sur le nombre d'épisodes
    rng, episode_rng = jax.random.split(rng) # On divise le générateur de nombres aléatoires pour obtenir une nouvelle clé pour l'épisode
    
    # 1. Jouer un épisode avec la politique actuelle
    trajectories = run_episode(env, params, episode_rng) # On simule un épisode dans l'environnement en utilisant la politique actuelle et on obtient les trajectoires (obs, action, reward)
    total_reward = sum([r for _, _, r in trajectories]) # On calcule la récompense totale en sommant les récompenses de toutes les étapes de l'épisode
    
    # 2. Mettre à jour les paramètres
    params = update_params(params, trajectories, lr=0.01)  # On met à jour les paramètres en utilisant les trajectoires de l'épisode et un taux d'apprentissage de 0.01
    
    if episode % 50 == 0: 
        print(f"Épisode {episode}, Récompense totale: {total_reward}")

env.close()