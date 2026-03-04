with open('Gym/JaxCode.py', encoding='utf-8') as f:
    content = f.read()

# 1. Supprimer watch_agent + bloc VIS UALISATION
import re
content = re.sub(
    r'# =============================================================================\n# VIS UALISATION.*?return rng',
    '',
    content,
    flags=re.DOTALL
)

# 2. Changer l'env d'entrainement en render_mode='human'
content = content.replace(
    "env = gym.make('CartPole-v1') # On cr\u00e9e l'environnement CartPole-v1 de gym",
    "# La fenetre reste ouverte pendant tout l'entrainement\nenv = gym.make('CartPole-v1', render_mode='human')  # fenetre visuelle active des le debut"
)

# 3. Supprimer VISUAL_CHECKPOINTS et le bloc watch_agent dans la boucle
content = content.replace(
    "VISUAL_CHECKPOINTS = {0, 249, 499}    # \u00e9pisodes o\u00f9 une fen\u00eatre CartPole s'ouvre\n",
    ""
)

# 4. Remplacer le bloc VISUAL_CHECKPOINTS dans la boucle par un sleep conditionnel
old_checkpoint_block = """    # Enregistrer la r\u00e9compense pour le graphique
    all_rewards.append(total_reward)
    # Ouvrir la fen\\u00eatre visuelle aux \\u00e9tapes cl\\u00e9s de l'entra\\u00eenement
    if episode in VISUAL_CHECKPOINTS:
        label = {0: "\\u00c9pisode 0 (agent vierge, agit au hasard)",
                 249: "\\u00c9pisode 250 (mi-entra\\u00eenement)",
                 499: "\\u00c9pisode 499 (fin d'entra\\u00eenement)"}[episode]
        rng = watch_agent(params, rng, label=label)"""

new_checkpoint_block = """    # Enregistrer la r\u00e9compense pour le graphique
    all_rewards.append(total_reward)"""

content = content.replace(old_checkpoint_block, new_checkpoint_block)

# 5. Modifier run_episode pour accepter un parametre slow
content = content.replace(
    "def run_episode(env, params, rng, verbose=False):",
    "def run_episode(env, params, rng, verbose=False, slow=False):"
)
content = content.replace(
    "        trajectories.append((obs, action, reward))\n        obs = next_obs",
    "        if slow:\n            time.sleep(0.04)  # ~25 fps quand slow=True\n        trajectories.append((obs, action, reward))\n        obs = next_obs"
)

# 6. Passer slow=True toutes les 10 episodes dans la boucle
content = content.replace(
    "    # 1. Jouer un \u00e9pisode avec la politique actuelle\n    trajectories = run_episode(env, params, episode_rng, verbose=verbose)",
    "    # 1. Jouer un \u00e9pisode avec la politique actuelle\n    # Ralentir toutes les 10 episodes pour bien voir l'agent evoluer\n    slow_render = (episode % 10 == 0)\n    trajectories = run_episode(env, params, episode_rng, verbose=verbose, slow=slow_render)"
)

# 7. Supprimer la demo finale (la fenetre est deja ouverte pendant tout l'entrainement)
demo_block_start = content.find('\n# =============================================================================\n# D\u00c9MO FINALE')
if demo_block_start != -1:
    # Garder seulement plt.show() a la fin
    content = content[:demo_block_start] + '\nplt.show()\nprint("[TERMINE] Ferme la fenetre du graphique pour quitter.")\n'

with open('Gym/JaxCode.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("OK - JaxCode.py mis a jour")
print("La fenetre CartPole s'ouvre au debut et reste ouverte tout l'entrainement.")
print("Toutes les 10 episodes : animation ralentie (~25fps) pour voir l'agent.")
print("Autres episodes : rapide (entrainement non bloque).")
