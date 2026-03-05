import gymnasium as gym
import matplotlib.pyplot as plt  # Pour tracer la courbe d'apprentissage
import matplotlib.patches as mpatches  # Pour les légendes du graphique
import time  # Pour les délais entre les frames (ralentir l'animation)
import os
from datetime import datetime

import jax  # On importe jax pour les calculs différentiables et les opérations sur GPU/TPU
import jax.numpy as jnp # On importe jax.numpy pour les opérations sur les tableaux, similaire à numpy mais optimisé pour jax


def print_jax_runtime_info():
    backend = jax.default_backend()
    devices = jax.devices()
    print("\n" + "="*60)
    print("[JAX RUNTIME]")
    print(f"  Backend actif : {backend}")
    print(f"  Devices       : {devices}")
    if backend != 'gpu':
        print("  ⚠️  GPU non utilisé : l'entraînement tournera sur CPU.")
    else:
        print("  ✅ GPU CUDA détecté : les opérations JAX sont envoyées au GPU.")
    print("="*60)

# =============================================================================
# STRATÉGIE UTILISÉE : REINFORCE (Monte-Carlo Policy Gradient)
# -----------------------------------------------------------------------------
# L'agent apprend une POLITIQUE STOCHASTIQUE (réseau linéaire) qui mappe
# directement les observations vers des probabilités d'action.
# À chaque épisode :
#   1. L'agent joue un épisode COMPLET avec la politique actuelle
#   2. On calcule les retours cumulés actualisés (Monte-Carlo)
#   3. On fait une descente de gradient pour maximiser les actions qui
#      ont mené à de bonnes récompenses (gradient de politique)
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


@jax.jit
def sample_action_core(rng, params, observation):
    logits = policy_network(params, observation)
    action = jax.random.categorical(rng, logits)
    return action, logits

# =============================================================================
# CRÉATION DE L'AGENT
# -----------------------------------------------------------------------------
# L'agent est défini par ses paramètres (poids 'w' et biais 'b').
# init_params() crée ces paramètres aléatoires → c'est la naissance de l'agent.
# Plus bas dans le code : params = init_params(rng)  ← instanciation réelle.
# =============================================================================
def init_params(rng, input_dim=4, output_dim=2): # On initialise les paramètres du réseau de politique 
    #On crée des paramètres aléatoires pour commencer
    k1, k2 = jax.random.split(rng) # On divise le générateur de nombres aléatoires pour obtenir deux clés
    w = jax.random.normal(k1, (input_dim, output_dim))* 0.1 # On initialise les poids avec une distribution normale et on les scale pour éviter des valeurs trop grandes
    b = jax.random.normal(k2, (output_dim,))*0.1
    params = {'w': w, 'b': b}
    print("\n" + "="*60)
    print("[AGENT] Naissance de l'agent !")
    print(f"  Entrées  : {input_dim} valeurs d'état (position, vitesse, angle, vitesse_angulaire)")
    print(f"  Sorties  : {output_dim} actions possibles (0=gauche, 1=droite)")
    print(f"  Poids W  : matrice {input_dim}x{output_dim}, initialisés aléatoirement")
    print(f"  Biais b  : vecteur de {output_dim}, initialisés aléatoirement")
    print("  → L'agent ne sait RIEN pour l'instant, il agit au hasard.")
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
    observation = jnp.asarray(observation, dtype=jnp.float32)
    # ACTION : tirage stochastique — la politique est probabiliste, pas déterministe
    action, logits = sample_action_core(rng, params, observation)
    if verbose:
        probs = jax.nn.softmax(logits)  # convertir logits → probabilités lisibles
        print(f"    [POLITIQUE] Logits bruts        : gauche={float(logits[0]):.3f}, droite={float(logits[1]):.3f}")
        print(f"    [POLITIQUE] Probabilités        : gauche={float(probs[0]):.1%}, droite={float(probs[1]):.1%}")
        print(f"    [ACTION]    Action choisie      : {'GAUCHE (0)' if int(action)==0 else 'DROITE (1)'}")
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
    step = 0  # compteur de pas

    if verbose:
        print("\n[ENV] Environnement réinitialisé.")
        print(f"  État initial → pos_chariot={obs[0]:.3f}, vit_chariot={obs[1]:.3f}, "
              f"angle_perche={obs[2]:.3f} rad, vit_angulaire={obs[3]:.3f}")
        print("  (Les valeurs sont proches de 0 : perche presque verticale, chariot au centre)")
        print("-"*60)

    while not done: # Tant que l'épisode n'est pas terminé
        rng, action_rng = jax.random.split(rng)
        step += 1
        log_this_step = verbose and step <= 4  # on log seulement les 4 premiers pas

        if log_this_step:
            print(f"\n  [PAS {step}]")
            print(f"    [ÉTAT]      pos={obs[0]:.3f}, vit={obs[1]:.3f}, "
                  f"angle={obs[2]:.3f} rad ({obs[2]*57.3:.1f}°), vit_ang={obs[3]:.3f}")

        # ACTION : l'agent consulte sa politique pour décider quoi faire
        action = sample_action(action_rng, params, obs, verbose=log_this_step)

        # appliquer l'action dans l'environnement
        # REWARD (RÉCOMPENSE) : +1 à chaque pas où la perche reste debout
        # next_obs = nouvel ÉTAT après l'action
        next_obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

        if log_this_step:
            print(f"    [REWARD]    Récompense reçue    : +{reward:.1f}  "
                  f"({'épisode TERMINÉ !' if done else 'perche encore debout ✓'})")
        if verbose and step == 4 and not done:
            print("    ... (logs masqués pour les pas suivants) ...")

        if slow:
            time.sleep(0.04)  # ~25 fps quand slow=True
        trajectories.append((obs, action, reward))
        obs = next_obs

    if verbose:
        print(f"\n[ENV] Épisode terminé après {step} pas.")
        print(f"  Récompense totale : {sum(r for _,_,r in trajectories):.1f}")
        print(f"  (Score max possible dans CartPole-v1 : 500)")

    return trajectories # On retourne la liste des trajectoires à la fin de l'épisode

# =============================================================================
# MISE À JOUR DE LA POLITIQUE — cœur de l'algorithme REINFORCE
# -----------------------------------------------------------------------------
# compute_loss() calcule la perte du gradient de politique :
#   loss = -Σ log π(a|s) * G   (G = retour cumulé actualisé)
# On minimise cette perte → l'agent augmente la probabilité des actions
# qui ont mené à des retours élevés (apprentissage par récompense différée).
# =============================================================================
def compute_loss(params, trajectories): # On définit une fonction pour calculer la perte à partir des paramètres et des trajectoires
    observations, actions, rewards = trajectories
    gamma = 0.99

    def discounted_scan(carry, reward):
        new_carry = reward + gamma * carry
        return new_carry, new_carry

    _, reversed_returns = jax.lax.scan(discounted_scan, 0.0, rewards[::-1])
    returns = reversed_returns[::-1]

    logits = observations @ params['w'] + params['b']
    log_probs = jax.nn.log_softmax(logits, axis=1)
    chosen_log_probs = log_probs[jnp.arange(actions.shape[0]), actions]
    return -jnp.mean(chosen_log_probs * returns)


# =============================================================================
# VARIANTE AVEC BASELINE — REINFORCE avec réduction de variance
# -----------------------------------------------------------------------------
# compute_loss_with_baseline() améliore REINFORCE en soustrayant une baseline
# (la moyenne des retours de l'épisode) à chaque retour cumulé.
# Avantage calculé : A = G - baseline
# Cela réduit la variance du gradient sans introduire de biais,
# car la baseline est indépendante de l'action choisie.
# Résultat : apprentissage plus stable et plus rapide qu'un REINFORCE pur.
# Note : cette version Python pur est moins optimisée que compute_loss (JAX).
# =============================================================================
def compute_loss_with_baseline(params, trajectories):
    total_loss = 0.0
    returns = []
    G = 0
    gamma = 0.99

    # Calcul des retours cumulés actualisés en parcourant l'épisode à l'envers
    # G_t = r_t + γ * G_{t+1}  (propagation des récompenses futures vers le passé)
    for _, _, reward in reversed(trajectories):
        G = reward + gamma * G
        returns.insert(0, G)  # on insère en tête pour maintenir l'ordre chronologique

    # Baseline = moyenne des retours de l'épisode
    # Sert de référence : une action est "bonne" si son retour dépasse cette moyenne
    baseline = sum(returns) / len(returns)

    for (obs, action, _), G in zip(trajectories, returns):
        # Calculer les logits et les log-probabilités via la politique
        logits = policy_network(params, obs)
        log_probs = jax.nn.log_softmax(logits)
        log_prob_action = log_probs[action]  # log-prob de l'action effectivement choisie

        # Avantage = écart entre le retour réel et la baseline
        # Si advantage > 0 : l'action était meilleure que la moyenne → on la renforce
        # Si advantage < 0 : l'action était pire que la moyenne → on la pénalise
        advantage = G - baseline
        total_loss += -log_prob_action * advantage  # gradient de politique pondéré

    # Moyenne sur tous les pas de temps pour normaliser selon la longueur de l'épisode
    return total_loss / len(trajectories)


def trajectories_to_arrays(trajectories):
    observations = jnp.asarray([obs for obs, _, _ in trajectories], dtype=jnp.float32)
    actions = jnp.asarray([int(action) for _, action, _ in trajectories], dtype=jnp.int32)
    rewards = jnp.asarray([reward for _, _, reward in trajectories], dtype=jnp.float32)
    return observations, actions, rewards

# Jax: gradient et mise à jour
#jax.grad transforme compute_loss en fonction qui retourne les gradients !
grad_loss = jax.jit(jax.grad(compute_loss)) # On utilise jax.grad pour obtenir une fonction qui calcule les gradients de la perte par rapport aux paramètres

def update_params(params, trajectories, lr=0.01, verbose=False):
    grads = grad_loss(params, trajectories)
    new_params = {
        'w': params['w'] - jnp.float32(lr) * grads['w'],
        'b': params['b'] - jnp.float32(lr) * grads['b']
    }
    if verbose:
        grad_norm_w = float(jnp.linalg.norm(grads['w']))
        grad_norm_b = float(jnp.linalg.norm(grads['b']))
        print(f"\n[MISE À JOUR POLITIQUE]")
        print(f"  Norme du gradient W : {grad_norm_w:.5f}")
        print(f"  Norme du gradient b : {grad_norm_b:.5f}")
        print(f"  Taux d'apprentissage : {lr}")
        print(f"  → Les poids sont ajustés pour favoriser les actions rentables.")
    return new_params
# INITIALISATION
# =============================================================================
rng = jax.random.PRNGKey(42) # On initialise une clé de générateur de nombres aléatoires pour jax
print_jax_runtime_info()

# CRÉATION DE L'ENVIRONNEMENT : CartPole-v1
# L'environnement simule une perche posée sur un chariot mobile.
# Objectif : maintenir la perche en équilibre le plus longtemps possible.
# La fenetre reste ouverte pendant tout l'entrainement
env = gym.make('CartPole-v1', render_mode='human')  # fenetre visuelle active des le debut

# CRÉATION DE L'AGENT : initialisation aléatoire des paramètres (poids + biais)
# C'est ici que l'agent "naît" — il ne sait encore rien, ses poids sont aléatoires.
params = init_params(rng) # On initialise les paramètres du réseau de politique

# ÉPISODES POUR LESQUELS ON AFFICHE LES DÉTAILS PAS-À-PAS
VERBOSE_EPISODES = {0, 100, 499}       # épisodes avec logs détaillés pas-à-pas

# Liste pour stocker les récompenses de chaque épisode → sert à tracer le graphique
all_rewards = []

n_episodes = 500
print("\n" + "="*60)
print("[DÉBUT ENTRAÎNEMENT]")
print(f"  Nombre d'épisodes : {n_episodes}")
print(f"  Algorithme        : REINFORCE (Monte-Carlo Policy Gradient)")
print(f"  Récompense        : +1 par pas de temps où la perche est debout")
print(f"  Objectif          : atteindre 500 pas (score max de CartPole-v1)")
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
    trajectory_arrays = trajectories_to_arrays(trajectories)
    params = update_params(params, trajectory_arrays, lr=0.01, verbose=verbose)

    # Enregistrer la récompense pour le graphique
    all_rewards.append(total_reward)
    if episode % 50 == 0 or verbose:
        print(f"\n» Épisode {episode:>3} | Pas : {n_steps:>4} | Récompense totale : {total_reward:>6.1f} "
              f"{'🎯 PARFAIT!' if total_reward >= 500 else ''}")

print("\n" + "="*60)
print("[FIN ENTRAÎNEMENT]")
print(f"  Dernier score : {total_reward:.1f} / 500")
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
output_dir = 'learning_curves'
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
curve_path = os.path.join(output_dir, f'learning_curve_{timestamp}.png')
plt.savefig(curve_path, dpi=120)  # Sauvegarde automatique (sans écraser les anciens runs)
plt.show(block=False)  # Affiche sans bloquer la suite du programme
print(f"[GRAPHIQUE] Courbe sauvegardée dans '{curve_path}' (fenêtre ouverte en arrière-plan)")

plt.show()
print("[TERMINE] Ferme la fenetre du graphique pour quitter.")
