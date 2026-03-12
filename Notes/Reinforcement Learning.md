C'est un paradigme d'un apprentissage automatique ou un agent autonome apprend à faire des décisions optimales et séquentielles en interagissant avec un environnement par essais et erreur

### Key Concepts:

Agent : The learner or decision-Maker

Environment: The world/system the agent intercat with
Action(A): The moves the agent can make
State(s): The current situation of the agent in the environment
Reward(R): Feedback from the environment indicating the success of any action.
Policy(pi): The strategy or rule set the agent uses to determine the next action.

### How it works ?

The agent observes the current state, select an action, perfoms it , and receive a reward while transitioning to a new state through repeated interactions, the agent updates its policy to favor actions that lead to higher long-term rewards

Exploration: explorer alternative
explotation: sticking to what work best.

#### Strategies:
Epsilon-greedy learning : 
On a une fonction qui genere des nombres aléatoires. Si le nb est plu spetit que certains seuils prédéfinies appelés epsilon . L'agent tente l'exploration sinon il restera dans l'exploitation.

Boltzmann exploration : Si l'agent continue à avoir des récompenses génagtives, il sera plu scusceptible d'explorer de nouvelles options Mais s'il a des récompenses positives, il restera dans l'exploitation.
#### Model-Based rainforcement learning 
L'agent crée un model de l'environnement . Le model rappresente des dynamiques de l'environnement(state transitions and reward probabilities).

L'agent utilise ce model pour planifier et evalue different actions avant de les faires dans le vrain environnement.

#### Model-free reinforcement learning
L'agent apprend tout seul dans son environnement réel, l'agent apprend l'état des la situation et actions ou l'optimisasion de la strategie avec la methode "trial and error".

#### Policy 
Defines the agent's behavior maps state for actions 
- Simples rules or complex computations
#### Value function :
Evaluates long-term benefits, not juste immediate rewards.
- Mesure l'opportunité d'un état qui envisage les résultats futures.
Exemple: Un véhicule peut éviter les manoeuvres imprudentes (gain à court terme ) pour maximiser la sécurité et l'efficacité globales

The process is mathematically framed ad a Markov Decision Process ([[MDP]]) where future state depend only on the current state and action, not on the prior sequence of events.
### Etapes
1. Policy(params,obs) -> action distribution
	Sans politique pas de décision 
	-> pas d'intercation avec l'environnement 
2. Compute_loss(params,batch) -> Scalar. 
	Sans fonction de perte, pas d'apprentissage -> l'agent ne s'améliore jamais 
3. Update - params(params, gradients) -> new params.
	Sans mise à jour les calculs loss ne servent à rien.
4. Collecte experience(env, policy, rng) -> trajectoires
	Sans données, rien à apprendre 
5. evalutate(env, policy, n_episodes -> mean reward)
	Sans évaluation, tu ne sais pas si ton agent apprend vraiment.
### Archictecture
``` python
# Initialisation
params = init.params()
rng= init.rng()
# Boucle principale d'entraînement 
for iteration in range (n_iterations)
#1. Collecter des données (fonction #4)
rajectoires= collect_experience(env,params.rng)
#Optionnel: stocker dans un replay buffer
replay_buffer.add(trajectoires)
#2. Calculer la loss (fonction #2)
loss= compute_loss (params,trajectories)
#3. Calculer les gradients
gradients= jax.grad(compute_loss) 
(params, trajectories)
#4. Mettre à jour les paramètres (fonction #3)
params= update_params(params, gradients)
#5. evaluer périodiquement(fonction #5)
if iteration % eval_frequency==0
	score= evaluate(env,params)
	print(f"Itération {iteration}, Score: {score})
	
```
### Les variations selon l'algo 


| Algo          | Particularité                          | Fonctions sup       |
| ------------- | -------------------------------------- | ------------------- |
| DQN           | Ajoute target_network et replay_buffer | update_target()     |
| Reinforcement | utilise les épisodes complets          | compute_returns()   |
| Actor-Critic  | Deux réseau (acteur + critique)        | compute_advantage() |
| SAC/DPPG      | Exploiration Complexe                  | exploration_noise() |
### Options avancées
- Replay buffer(utile pour DQN)
- Target networks, Normalization
- Multiple environments
- Logging complexe

## Mujoco

You define robot and environment in MJCF define a reward, and let an RL agent train in Python

