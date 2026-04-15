Voici le plan complet, chaque étape ne commence que si la précédente est validée.

---

## Point Z (l'objectif final)

> L'humain dit "assemble moi ça". Le robot voit 3 objets, le LLM décompose en sous-tâches, TD-MPC2 exécute chaque pick/place/insert, le tout est logué en base. Succès > 50% sur les primitives.

---

## Le plan A → Z

### A — Scène MuJoCo fonctionnelle
**Quoi :** `scene_assembly.xml` charge sans erreur avec UR5e + Robotiq 2F-85 + 3 objets + zones cibles.
**Critère de succès :** `mujoco.MjModel.from_xml_path()` → `nq`, `nv`, `nu` corrects, simulation step OK, TCP et pinch positions raisonnables.
**Bloqué par :** rien (on commence ici)

### B — Environnement Gymnasium basique
**Quoi :** `env.py` réécriture complète. State 16-dim, action 7-dim (6 joints + gripper), reset, step, reward basique (juste reach = -distance).
**Critère de succès :** `env.reset()` + 100× `env.step(random_action)` sans crash. Vérifier que le gripper s'ouvre/ferme, que les objets restent sur la table.
**Bloqué par :** A

### C — Config + World Model adaptés aux nouvelles dimensions
**Quoi :** `config.py` (state_dim=16, action_dim=7, goal_dim=10). `world_model.py` adapté. Supprimer `goal_encoder.py` et `policy.py`.
**Critère de succès :** instancier `WorldModel(cfg)`, forward pass avec tenseurs aux bonnes dimensions, `compute_loss()` retourne des gradients.
**Bloqué par :** B

### D — Primitive `reach` entraînée
**Quoi :** Le bras apprend à amener le TCP vers une position 3D aléatoire dans l'espace de travail.
**Critère de succès :** Après entraînement (~2000 épisodes), distance finale TCP→cible < 2cm dans **80%** des cas.
**Bloqué par :** C

### E — Primitive `pick` entraînée
**Quoi :** Le bras apprend à s'approcher d'un objet, fermer le gripper, et le soulever.
**Critère de succès :** Objet soulevé > 5cm au-dessus de la table dans **50%** des cas.
**Bloqué par :** D (réutilise le reach appris comme initialisation)

### F — Primitive `place` entraînée
**Quoi :** Le bras tient un objet et le dépose sur une zone cible.
**Critère de succès :** Distance objet→cible < 3cm après lâcher dans **50%** des cas.
**Bloqué par :** E (doit savoir pick pour commencer place avec objet en main)

### G — Primitive `insert` entraînée
**Quoi :** Le bras insère un peg dans un slot avec précision.
**Critère de succès :** Peg dans le slot (distance < 5mm) dans **30%** des cas. (C'est dur, 30% est ambitieux.)
**Bloqué par :** F

### H — Scene Parser
**Quoi :** `scene_parser.py` lit les positions de tous les objets et zones depuis `MjData` et retourne un dict structuré.
**Critère de succès :** Retourne `{"red_block": [x,y,z], "slot_B": [x,y,z], ...}` correct à chaque pas.
**Bloqué par :** A (indépendant de D-G, peut être fait en parallèle)

### I — Task Planner (LLM)
**Quoi :** `task_planner.py` envoie instruction + objets au LLM, reçoit une séquence de primitives structurée.
**Critère de succès :** `"assemble moi ça"` + 3 objets → liste de primitives parseable. Testé sur 10 instructions différentes.
**Bloqué par :** H (a besoin de la liste d'objets)

### J — Pipeline complet `run.py`
**Quoi :** Boucle complète : instruction humaine → scene parser → LLM → exécution séquentielle des primitives → replanification si échec → log SQLite.
**Critère de succès :** Bout-en-bout fonctionne sur 3 scénarios :
1. "mets le cube rouge dans la zone jaune" → pick + place
2. "insère le peg vert dans le slot" → pick + insert
3. "assemble tout" → LLM décompose → séquence complète

**Bloqué par :** G + I (tout doit être prêt)

---

## Résumé visuel

```
A ──→ B ──→ C ──→ D ──→ E ──→ F ──→ G ──┐
                                           ├──→ J (pipeline complet) = Z
A ──→ H ──→ I ───────────────────────────┘
```

**Chemin critique :** A → B → C → D → E → F → G → J
**Chemin parallèle :** A → H → I (peut être fait pendant D-G)

---

## Estimation réaliste de difficulté

| Étape | Difficulté | Risque d'échec |
|---|---|---|
| A — XML scène | Facile | Faible |
| B — Env Gymnasium | Moyen | Faible |
| C — Config + WM dims | Facile | Faible |
| D — Reach | Facile | Faible |
| E — Pick | **Moyen-Dur** | Moyen (le grasp est le point dur) |
| F — Place | Moyen | Faible (si pick marche, place suit) |
| G — Insert | **Dur** | **Élevé** (précision sub-mm) |
| H — Scene Parser | Facile | Faible |
| I — Task Planner | Facile | Faible |
| J — Pipeline | Moyen | Moyen (intégration) |

**Les 2 points de risque : E (pick) et G (insert).** Si pick ne marche pas, tout le reste tombe. Si insert ne marche pas, on peut quand même démontrer pick+place.

---

On attaque A ?