States : $S$ 
Model: $T(S, a, S') ~ P(S'|S,a)$
Actions: $A(S) , A$
Reward: $R(S), R(S,a), R(S,a,S')$

-------------------------------------------------------------------------
Policy: $π(S)$  -> a 
	   $π$*



1. S: A position on a grid being a cell (1,1).
2. A: Move UP, Down, : Each state can have one or more possible actions
3. Transition Model(T) : The model tells us what happens when an action is taken in a state. it's like asking : "if move right from here, where will I land?" Sometimes the outcome isn't always the same that's uncertainty. 
		For exemple : 
		- 80% chance of moving in the intend direction
		- 10% chance of slipping to the left
		- 10% chance of slipping to the right?
	This randomness is called a stochastic transition

4. Reward(R) : +1 for reaching the goal 
			 -1 for stepping into fire
				-0.1 for each step to encourage fewer moves

5. Policy(π): A policy is the agent's plan. It tells the agent: "If you are in this state, take this action" The goal is to find the best policy that helps the agent earn the highest total reward over time. 