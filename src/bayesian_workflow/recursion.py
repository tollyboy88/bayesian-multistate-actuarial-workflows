import numpy as np


def terminal_distribution(initial: np.ndarray, matrices: list[np.ndarray]) -> np.ndarray:
    distribution = np.asarray(initial, dtype=float)
    for matrix in matrices:
        distribution = distribution @ np.asarray(matrix, dtype=float)
    return distribution


def monotonicity_check(base: list[np.ndarray], improved: list[np.ndarray], error_index: int = 1) -> bool:
    initial = np.zeros(base[0].shape[0]); initial[error_index] = 1.0
    return terminal_distribution(initial, improved)[error_index] <= terminal_distribution(initial, base)[error_index] + 1e-12
