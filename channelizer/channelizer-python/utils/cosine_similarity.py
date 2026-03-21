import numpy as np

def cosine_similarity(signals_1: np.ndarray, signals_2: np.ndarray, noise_floor: float = 1e-3) -> np.ndarray:
    norm_1 = np.linalg.norm(signals_1, ord=2, axis=1)
    norm_2 = np.linalg.norm(signals_2, ord=2, axis=1)
    dot_product = np.abs(np.sum(signals_1 * np.conj(signals_2), axis=-1))

    epsilon = 1e-12 
    cos_sim = dot_product / (norm_1 * norm_2 + epsilon)
    
    return cos_sim