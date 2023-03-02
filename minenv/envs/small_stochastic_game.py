import minenv
import numpy as np

from numpy.typing import NDArray
from typing import Dict, Optional, Set, Tuple

ActType = int
ObsType = int

class SmallStochasticGame(minenv.SimpleEnvironment[ActType, ObsType]):

	def __init__(
		self,
		rewards: Dict[int, NDArray[Tuple[np.float64, ...]]],
		init_dist: Optional[NDArray[np.float64]] = None,
		transitions: Optional[Dict[int, NDArray[np.float64]]] = None,
		final_state: Optional[int] = None,
		rng: Optional[np.random._generator.Generator] = None
	):
		r'''
		Initialize a (small) stochastic game.

		Parameters:
			rewards (Dict[int, NDArray[Tuple[np.float64, ...]]]): A dictionary of reward structures, one for each state.
			init_dist (NDArray[np.float64], optional): An initial distribution over the states for resetting the environment.
			transitions (Dict[int, NDArray[np.float64]], optional): A dictionary of transition probabilities.
			final_state (int, optional): A final state, which ends the game if entered.
			rng (np.random._generator.Generator, optional): A random number generator.
		'''
		super(SmallStochasticGame, self).__init__(rng=rng)

		self.rewards = rewards

		self._states: NDArray[np.int64] = np.fromiter(self._rewards.keys(), dtype=np.int64)
		self._n_states = len(self._states)
	
		# validate reward structure
		shapes = set(len(matrix.shape) for matrix in rewards.values())
		if len(shapes)>1:
			raise ValueError('All reward matrices must have the same dimension.')
		self._n_agents: int = shapes.pop()

		self.transitions = transitions

		self.init_dist = init_dist

		self.final_state = final_state

	@property
	def rewards(self):
		return self._rewards

	@rewards.setter
	def rewards(self, rewards: Dict[int, NDArray[Tuple[np.float64, ...]]]):
		self._rewards = rewards
	
	@property
	def init_dist(self):
		return self._init_dist

	@init_dist.setter
	def init_dist(self, init_dist: Optional[NDArray[np.float64]] = None):
		self._init_dist: NDArray[np.float64] = np.full_like(self._states, 1/self._n_states) if init_dist is None else init_dist

	@property
	def transitions(self):
		return self._transitions

	@transitions.setter
	def transitions(self, transitions: Optional[Dict[int, NDArray[np.float64]]] = None):
		self._transitions = transitions

		# validate state transition structure
		if self._transitions is None:
			if self._n_states>1:
				raise ValueError('There is more than one state but no transition structure.')
			self._transitions = {}
			for state in self._states:
				self._transitions.update({state: np.ones((*self._rewards[state].shape, self._n_states))})

		if not np.all(np.fromiter(self._transitions.keys(), dtype=np.int64)==self._states):
			raise ValueError(f'The states in the transition structure must equal {self._states}.')

		bad_states: Set[int] = set()
		for state in self._transitions.keys():
			if self._transitions[state].shape[-1]!=self._n_states:
				# the number of weights must equal the number of states
				bad_states.add(state)
			
			elif not (np.all(self._transitions[state]>=np.array(0)) and (np.all(np.sum(self._transitions[state], axis=-1)>np.array(0)))):
				# the weights must be non-negative and have positive sum for each state and action profile
				bad_states.add(state)
			
			elif self._transitions[state].shape[:-1]!=self._rewards[state].shape:
				bad_states.add(state)
				
		if len(bad_states)>0:
			raise ValueError(f'The states in {bad_states} do not have {self._n_states} non-negative weights for every possible action profile.')

		# normalize weights to sum to 1
		for state in self._transitions.keys():
			self._transitions[state] = np.divide(self._transitions[state], np.sum(self._transitions[state], axis=-1, keepdims=True))

	@property
	def final_state(self):
		return self._final_state

	@final_state.setter
	def final_state(self, final_state: Optional[int] = None):
		self._final_state = final_state
		# validate final state
		if (self._final_state is not None) and (self._final_state not in self._states):
			raise ValueError(f'The final state, if given, must be in {self._states}.')

	@property
	def states(self):
		return self._states

	@property
	def n_agents(self):
		return self._n_agents

	@property
	def n_states(self):
		return self._n_states

	def reset(self) -> Tuple[ObsType, Tuple[Set[ActType], ...]]:
		super(SmallStochasticGame, self).reset()
		
		self._state = self._rng.choice(self._states, p=self._init_dist)
		return self._state, self._action_sets(self._state)

	def step(self, actions: Tuple[int, ...]) -> Tuple[ObsType, Tuple[Set[ActType], ...], Tuple[float, ...], bool]:
		super(SmallStochasticGame, self).step(actions)

		if len(actions)!=self._rewards[self._state].ndim:
			raise ValueError(f'Exactly {self._rewards[self._state].ndim} actions are required but {len(actions)} were given.')
		elif not all(i<j for i,j in zip(actions, self._rewards[self._state].shape)):
			raise ValueError(f'The actions must be bounded from above by {self._rewards[self._state].shape}.')

		rewards = self._rewards[self._state][actions]
		if self._transitions is not None:
			self._state = self._rng.choice(self._states, p=self._transitions[self._state][actions])
		return self._state, self._action_sets(self._state), rewards, (self._state==self._final_state) # all agents can fully observe the state

	def _action_sets(self, state: int) -> Tuple[Set[ActType], ...]:
		reward_shape = self._rewards[self._state].shape
		return tuple(set(range(reward_shape[idx])) for idx in range(len(reward_shape)))

class MatrixGame(SmallStochasticGame):

	def __init__(self, rewards: NDArray[Tuple[np.float64, ...]], rng: Optional[np.random._generator.Generator] = None):
		super(MatrixGame, self).__init__({1: rewards}, rng=rng)
