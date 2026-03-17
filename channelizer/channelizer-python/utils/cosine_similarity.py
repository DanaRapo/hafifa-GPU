import numpy as np

def cosine_similarity(signals_1: np.ndarray, signals_2: np.ndarray) -> np.ndarray:
    norm_1 = np.linalg.norm(signals_1, ord=2, axis=1)
    norm_2 = np.linalg.norm(signals_2, ord=2, axis=1)
    return np.real(np.sum(signals_1 * np.conj(signals_2), axis=1)) / (norm_1 * norm_2 + 1e-8)