import minenv
import numpy as np

# from math import prod
from numpy.typing import NDArray
from typing import Dict, Generator, Literal, Optional, Set, Tuple, Union

ColType = Literal['b', 'r'] # color type ('blue', 'red')
ActType = Literal['D', 'L', 'R', 'U'] # action type ('down', 'left', 'right', or 'up')
ObsType = NDArray[np.int64] # observation type (an array giving local views of the grid)

class CoinGame(minenv.SimpleEnvironment[ActType, ObsType]):

	COLORS: Tuple[ColType, ColType] = ('b', 'r')
	GRID_ACTIONS: Dict[ActType, NDArray[np.int64]] = {'D': np.array([0, -1], dtype=np.int64), 'L': np.array([-1, 0], dtype=np.int64), \
		'R': np.array([1, 0], dtype=np.int64), 'U': np.array([0, 1], dtype=np.int64)}
	
	def __init__(
		self,
		grid_shape: Tuple[int, int] = (5, 5),
		n_coins: int = 1,
		obs_window: Tuple[int, int] = None,
		coin_payoffs : NDArray[np.float64] = np.array([[1, 0], [1, -2]], dtype=np.float64),
		rng: Optional[np.random._generator.Generator] = None
	):
		r'''
		Initialize a grid-based coin game.

		Parameters:
			grid_shape (Tuple[int, int]): Shape of the game grid (horizontal, vertical).
			n_coins (int): Total number of coins on the grid.
			obs_window (Tuple[int, int]): Size of the observation window (steps left/right and steps up/down).
			coin_payoffs (NDArray[np.float64]): A 2x2 matrix giving the payoffs for picking up coins of both colors.
			rng (np.random._generator.Generator, optional): A random number generator.
		'''
		super(CoinGame, self).__init__(rng=rng)
		
		self._grid_shape = grid_shape
		self._locations: NDArray[np.int64] = np.arange(np.prod(self._grid_shape), dtype=np.int64)

		self._agent_pos: Dict[ColType, NDArray[np.int64]] = {}

		self._n_coins = n_coins
		print("Placing blue and red coin alternately on the grid (instead of randomly choosing coin color)")

		# two agents can be at the same location but two coins cannot
		if len(self._locations)<1+self._n_coins:
			raise ValueError(f'There are {len(self._locations)} spots on the grid but {1}+{self._n_coins}={1+self._n_coins} spots needed.')

		self._obs_window: Tuple[int, int] = grid_shape if obs_window is None else obs_window

		self._coin_payoffs = coin_payoffs
		self.current_coin_color = self._rng.integers(2, 4) # maintain a memory of which color coin was generated in last step

	def reset(self) -> Tuple[Tuple[ObsType, ObsType], Set[ActType]]:
		super(CoinGame, self).reset()

		self._state = np.zeros((4, *self._grid_shape), dtype=int)

		# set locations of blue and red agents, storing positions as arrays to allow addition based on grid actions
		pos_generator = self._sample_from_grid(size=2, replace=True)
		self._agent_pos.update({color: np.array(next(pos_generator)) for color in CoinGame.COLORS})

		for idx, color in enumerate(CoinGame.COLORS):
			self._state[(idx, *self._agent_pos[color])] = 1

		# set locations of coins (non-overlapping and distinct from those of the agents)
		self._place_coins(self._n_coins)

		return self._get_observation(), set(CoinGame.GRID_ACTIONS.keys())

	def step(self, actions: Tuple[ActType, ...]) -> Tuple[Tuple[ObsType, ObsType], Set[ActType], Tuple[float, float], bool]:
		super(CoinGame, self).step(actions)

		if len(actions)!=2:
			raise ValueError(f'Exactly {2} actions are required but {len(actions)} were given.')

		# agents move simultaneously (also why agent positions are arrays instead of tuples)
		for idx, color in enumerate(CoinGame.COLORS):
			self._state[(idx, *self._agent_pos[color])] = 0
			self._agent_pos.update({color: (self._agent_pos[color]+CoinGame.GRID_ACTIONS[actions[idx]]) % self._grid_shape})
			self._state[(idx, *self._agent_pos[color])] = 1

		b_reward, r_reward = 0, 0
		coins_collected = 0
		if np.array_equal(self._agent_pos['b'], self._agent_pos['r']):

			# whether the blue agent wins (if both agents try to pick up a coin simultaneously)
			b_wins = self._rng.choice([False, True])

			if self._state[(2, *self._agent_pos['b'])]==1:
				# blue coin
				b_reward += b_wins*self._coin_payoffs[0, 0] + (1-b_wins)*self._coin_payoffs[1, 1]
				r_reward += b_wins*self._coin_payoffs[0, 1] + (1-b_wins)*self._coin_payoffs[1, 0]
				coins_collected += 1
				self._state[(2, *self._agent_pos['b'])] = 0

			elif self._state[(3, *self._agent_pos['b'])]==1:
				# red coin
				b_reward += b_wins*self._coin_payoffs[1, 0] + (1-b_wins)*self._coin_payoffs[0, 1]
				r_reward += b_wins*self._coin_payoffs[1, 1] + (1-b_wins)*self._coin_payoffs[0, 0]
				coins_collected += 1
				self._state[(3, *self._agent_pos['b'])] = 0
		else:
			# check if blue player picks up a coin
			
			if self._state[(2, *self._agent_pos['b'])]==1:
				# collect blue coin
				coins_collected += 1
				self._state[(2, *self._agent_pos['b'])] = 0
				b_reward += self._coin_payoffs[0, 0]
				r_reward += self._coin_payoffs[0, 1]

			elif self._state[(3, *self._agent_pos['b'])]==1:
				# collect red coin
				coins_collected += 1
				self._state[(3, *self._agent_pos['b'])] = 0
				b_reward += self._coin_payoffs[1, 0]
				r_reward += self._coin_payoffs[1, 1]

			# check if red player picks up a coin

			if self._state[(2, *self._agent_pos['r'])]==1:
				# collect blue coin
				coins_collected += 1
				self._state[(2, *self._agent_pos['r'])] = 0
				b_reward += self._coin_payoffs[1, 1]
				r_reward += self._coin_payoffs[1, 0]
				
			elif self._state[(3, *self._agent_pos['r'])]==1:
				# collect red coin
				coins_collected += 1
				self._state[(3, *self._agent_pos['r'])] = 0
				b_reward += self._coin_payoffs[0, 1]
				r_reward += self._coin_payoffs[0, 0]

		if coins_collected>0:
			self._place_coins(coins_collected)

		return self._get_observation(), set(CoinGame.GRID_ACTIONS.keys()), (b_reward, r_reward), False

	def _get_observation(self) -> Tuple[ObsType, ObsType]:
		observation = {}
		for color in CoinGame.COLORS:
			ranges = {}
			for grid_dim in range(2):
				if 1+2*self._obs_window[grid_dim]>self._grid_shape[grid_dim]:
					current_axis_range = range(self._grid_shape[grid_dim])
				else:
					current_axis_range = range(self._agent_pos[color][grid_dim]-self._obs_window[grid_dim], 1+self._agent_pos[color][grid_dim]+self._obs_window[grid_dim])
				ranges.update({grid_dim+1: current_axis_range})
			observation.update({color: self._state.take(ranges[1], axis=1, mode='wrap').take(ranges[2], axis=2, mode='wrap')})
		return (observation['b'], observation['r'])

	def _place_coins(self, n_coins: int):
		for coin_idx in self._sample_from_grid(size=n_coins, p=1-np.max(self._state, axis=0)):
			# assign each coin a color, blue or red, each with probability 1/2
			self._state[(self.current_coin_color, *coin_idx)] = 1
			if self.current_coin_color == 2:
				self.current_coin_color = 3
			elif self.current_coin_color == 3:
				self.current_coin_color = 2
			else:
				raise ValueError(f'Last coin was {self.current_coin_color}')
			# print(f'Placed coin at {coin_idx} with color {self.current_coin_color}')

	def _sample_from_grid(self, size: Optional[int] = None, replace: bool = False, p: Optional[NDArray[np.float64]] = None) -> Generator[Tuple[int, ...], None, None]:
		if p is not None:
			p = p.ravel(order='C')/np.sum(p)
		return (np.unravel_index(idx, self._grid_shape, order='C') for idx in self._rng.choice(self._locations, size=size, replace=replace, p=p))
