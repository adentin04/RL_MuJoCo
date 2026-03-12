1. Théorie RL : MDP(états, Actions, Récompenses)
2. Implémentation: Classe UR5eReachEnv.
	- state = observation(positions + vitesses + cible)
	- action = commandes joints
	- reward = f(distance, mouvement)
3. Algorithme: SAC (Stable Baseline 3)
	- Apprend politique π(a|S)
	- Maximise récompense cumulative.
4. Résultat: Le bras apprend à atteindre la cible 