Agent -> SAC( l'algorithme )
Environnement ->la classe
état(state) -> observation(positions + vitesses + cible)
Action -> commandes articulaires[-1,1] -> positions joints récompense -> _compute_reward()_ 
(distance + bonus)
Politique(Policy) -> réseau de neurones dans sac.
