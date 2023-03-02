import numpy as np

from numpy.typing import NDArray
from typing import Generator, Tuple

def compress_matrices(matrices: Tuple[NDArray[np.float64], ...]) -> NDArray[Tuple[np.float64, ...]]:
	r'''
	Converts a tuple of multidimensional arrays into a single multidimensional array of tuples.

	Parameters:
		matrices (Tuple[NDArray[np.float64], ...]): A tuple of multidimensional arrays, all of the same shape.

	Returns:
		NDArray[Tuple[np.float64, ...]]: A multidimensional array of tuples, giving a well-defined payoff structure for a state.
	'''
	# validate payoff matrices
	if len(set([matrix.shape for matrix in matrices]))>1:
		raise ValueError('All matrices must have the same shape.')
	FloatTuple = ', '.join(['f' for _ in range(len(matrices))])
	return np.reshape(np.fromiter(map(lambda idx: tuple([matrix[idx] for matrix in matrices]), \
		np.ndindex(matrices[0].shape)), dtype=FloatTuple), matrices[0].shape)

def extract_matrices(payoffs: NDArray[Tuple[np.float64, ...]], indices: Optional[Tuple[int, ...]] = None) -> Generator[Tuple[NDArray[np.float64], ...], None, None]:
	r'''
	TODO: comment here

	Parameters:
		payoffs (): 

		indices (): 

	Returns:
		Generator[Tuple[NDArray[np.float64], ...], None, None]: 
	'''
	indices = range(len(payoffs.flat[0])) if indices is None
	return (np.fromiter(map(lambda x: payoffs[x][idx], \
		np.ndindex(payoffs.shape)), dtype=np.float64).reshape(payoffs.shape) for idx in indices)
	