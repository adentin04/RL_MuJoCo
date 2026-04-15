# Architecture — Goal-Conditioned TD-MPC2 avec LLM Planner

## Type de projet

**Goal-Conditioned Model-Based Reinforcement Learning pour l'assemblage robotique.**

Le robot apprend par essai-erreur en simulation (pas de démonstrations humaines) à exécuter des tâches d'assemblage décrites en langage naturel.

---

## Pipeline complet

```
Humain: "assemble moi ça"
        ↓
   [1] Scene Parser: lit les positions des objets depuis MuJoCo
        → {red_block: [0.35, -0.15, 0.065], blue_block: [...], green_peg: [...]}
        ↓
   [2] Task Planner (LLM via API): reçoit instruction + liste objets
        → [pick(green_peg), insert(green_peg, slot_B), pick(red_block), place(red_block, zone_A)]
        ↓
   [3] Pour CHAQUE primitive:
        goal = [type_one_hot(4), object_pos(3), target_pos(3)]  → 10-dim
        ↓
        TD-MPC2 World Model + MPPI → action (7-dim) à 50Hz
        ↓
        MuJoCo exécute → nouvel état → reward
        ↓
        Succès? → primitive suivante
        Échec?  → LLM replanifie
        ↓
   [4] SQLite log le résultat
```

---

## Stack technique

| Couche | Outil | Rôle | Tourne sur |
|---|---|---|---|
| **Simulation** | MuJoCo 3.6 | Physique, contacts, rendu | CPU |
| **Robot** | UR5e + Robotiq 2F-85 | 6 DOF + gripper | (XML) |
| **Raisonnement** | LLM via API (Claude/GPT) | Décompose instruction → primitives | Cloud |
| **Apprentissage** | TD-MPC2 (PyTorch) | World model latent | GPU (6GB) |
| **Planification** | MPPI (512 trajectoires) | Choisit la meilleure action | GPU |
| **Mémoire** | Replay Buffer | Stocke transitions pour entraînement | RAM |
| **Logs** | SQLite | Succès/échec par épisode | Disque |

---

## Les 4 primitives entraînées par TD-MPC2

| Primitive | State | Goal | Reward | Succès si |
|---|---|---|---|---|
| `reach(pos)` | joints(6) + gripper(1) + tcp(3) | target_pos(3) | -distance(tcp, target) | distance < 2cm |
| `pick(obj)` | joints(6) + gripper(1) + tcp(3) + obj_pos(3) | obj_pos(3) | -dist + grasp_bonus | objet soulevé > 5cm |
| `place(obj, target)` | joints(6) + gripper(1) + tcp(3) + obj_pos(3) | target_pos(3) | -dist(obj, target) | objet sur target < 3cm |
| `insert(obj, hole)` | joints(6) + gripper(1) + tcp(3) + obj_pos(3) | hole_pos(3) | -dist + alignment_bonus | objet inséré < 2mm |

### Dimensions

- **State** = 16-dim (6 joints + 1 gripper + 3 tcp + 3 objet + 3 cible)
- **Action** = 7-dim (6 joints + 1 gripper)
- **Goal** = 10-dim (4 one-hot type + 3 obj_pos + 3 target_pos)

---

## Structure de fichiers

```
llm/
├── ARCHITECTURE.md        ← ce document
├── config.py              ← hyperparamètres (state_dim=16, action_dim=7, goal_dim=10)
├── world_model.py         ← encoder, dynamics, reward, value, policy
├── planner.py             ← MPPI (512 trajectoires, 64 élites, horizon 5)
├── replay_buffer.py       ← stocke transitions, sample chunks
├── experience_db.py       ← SQLite logs
├── task_planner.py        ← LLM API → liste de primitives
├── scene_parser.py        ← lit positions objets depuis MuJoCo
├── env.py                 ← 4 types de reward, state 16-dim, action 7-dim
├── train.py               ← entraîne les 4 primitives
└── run.py                 ← pipeline complet (input humain → exécution)

universal_robots_ur5e/
├── scene_assembly.xml     ← UR5e + Robotiq 2F-85 + objets d'assemblage
├── ur5e.xml
└── robotiq_2f85/
```

---

## Ce qu'on NE fait PAS

- ❌ Pas de VLA (Vision-Language-Action model)
- ❌ Pas de SBERT
- ❌ Pas de CNN pour la vision (en sim, état MuJoCo suffit)
- ❌ Pas de fine-tuning de modèle pré-entraîné
- ❌ Pas de démonstrations humaines
- ❌ Pas de sim-to-real (pour l'instant)

---

## Ordre de construction

```
1. Fixer scene_assembly.xml (Robotiq 2F-85)
2. Renommer vla/ → llm/, nettoyer les fichiers obsolètes
3. Adapter config.py (nouvelles dimensions)
4. Réécrire env.py (4 primitives, rewards, state 16-dim)
5. Adapter world_model.py (nouvelles dimensions)
6. Créer scene_parser.py
7. Créer task_planner.py (LLM API)
8. Créer run.py (pipeline complet)
9. Entraîner reach → pick → place → insert
10. Test bout-en-bout
```

---

## Justification du choix LLM + TD-MPC2

| Contrainte | VLA | LLM + TD-MPC2 |
|---|---|---|
| Pas de démonstrations | ❌ Bloquant | ✅ Exploration autonome |
| 6GB VRAM | ⚠️ Limité | ✅ OK |
| Précision insertion | ❌ Open-loop | ✅ MPPI replanifie chaque pas |
| Généralisation | ✅ | ✅ (via LLM) |
| Complexité code | Simple | Moyenne |
