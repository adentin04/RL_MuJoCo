# Super Mario A3C — Explication du fonctionnement du code

## 1) Vue d’ensemble
Le projet **Super_Mario_A3C_AI_vs_Human** contient deux usages principaux :

1. **Jouer / comparer IA vs humain** via une interface Pygame (`test.py`).
2. **Entraîner l’agent IA** avec l’algorithme **A3C** (`train.py` + `src/process.py`).

---

## 2) Point d’entrée (mode jeu)
Le script recommandé est :

- `run_game.sh`

Ce script :
- crée/active un environnement virtuel `.venv`,
- installe les dépendances (`requirements.txt`),
- installe PyTorch CPU,
- lance `python test.py`.

Le fichier `README_SETUP.md` décrit cette procédure.

---

## 3) Fichier principal du mode IA vs Humain
Le cœur du jeu interactif est :

- `test.py`

### Ce que fait `test.py`
- Initialise deux environnements Mario :
  - un pour l’IA (`create_train_env`),
  - un pour l’humain (`create_human_env`).
- Ouvre une interface Pygame avec :
  - boutons de sélection de modèle (épisode + taille de canaux),
  - bouton start/exit,
  - affichage des deux parties en parallèle.
- Charge un modèle entraîné (`trained_models/...`) selon le bouton choisi.
- Pilote l’humain via joystick (si détecté) sinon clavier.
- Enregistre un score/temps utilisateur dans `ranking.txt`.

---

## 4) Prétraitement de l’environnement
Dans `src/env.py`, le pipeline est :

1. Création de l’env Mario (`gym_super_mario_bros.make`).
2. Mapping des actions avec `JoypadSpace` (actions discrètes).
3. `CustomReward` :
   - passage RGB -> niveaux de gris,
   - redimensionnement en **84x84**,
   - shaping de reward à partir du score,
   - bonus/malus de fin (drapeau ou échec).
4. `CustomSkipFrame` : empilement de 4 frames (state temporel).

Résultat : l’état réseau est un tenseur de frames empilées, plus compact et informatif.

---

## 5) Modèle IA (Actor-Critic)
Dans `src/model.py`, classe `ActorCritic` :

- 4 couches convolutionnelles (extraction visuelle),
- 1 `LSTMCell` (mémoire temporelle),
- 2 têtes de sortie :
  - **Actor** (`actor_linear`) : logits des actions,
  - **Critic** (`critic_linear`) : valeur d’état.

Le paramètre `BIT` contrôle le nombre de canaux conv (ex: 16/32/64/128).

---

## 6) Entraînement A3C
Les fichiers principaux :
- `train.py`
- `src/process.py`
- `src/optimizer.py`

### `train.py`
- Lit les arguments (world, stage, lr, gamma, etc.).
- Crée le modèle global partagé (`global_model`).
- Crée l’optimiseur partagé `GlobalAdam`.
- Lance l’entraînement local (`local_train`) puis le test (`local_test`).

### `src/process.py` — logique A3C
- `local_train(...)` :
  - joue des séquences de pas (`num_local_steps`),
  - calcule policy + value,
  - échantillonne les actions,
  - accumule rewards/log-probs/entropie,
  - calcule les pertes actor/critic + régularisation entropie,
  - pousse les gradients vers le modèle global,
  - sauvegarde périodiquement les poids.
- `local_test(...)` :
  - utilise la politique gloutonne (`argmax`) pour évaluer l’agent.

### `src/optimizer.py`
`GlobalAdam` met ses états (`step`, `exp_avg`, `exp_avg_sq`) en mémoire partagée, utile pour le multi-process A3C.

---

## 7) Chargement des modèles dans l’UI
Dans `test.py`, un dictionnaire relie chaque bouton à :
- le chemin du modèle (`trained_models/<episode>_<bit>_1_1`),
- la valeur `bit_size` (16/32/64/128),
- un texte d’affichage.

Au clic :
- instanciation du réseau avec le bon `bit_size`,
- chargement des poids `a3c_super_mario_bros_<world>_<stage>`,
- activation du lancement via bouton Start.

---

## 8) Résumé très court
- `run_game.sh` : setup + lancement.
- `test.py` : interface IA vs humain + sélection de modèles.
- `src/env.py` : prétraitement observations + reward shaping.
- `src/model.py` : CNN + LSTM + Actor-Critic.
- `train.py` / `src/process.py` : boucle d’entraînement A3C.
- `ranking.txt` : score/temps humain sauvegardé.
