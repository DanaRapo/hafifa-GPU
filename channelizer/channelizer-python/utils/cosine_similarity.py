import numpy as np

def cosine_similarity(signals_1: np.ndarray, signals_2: np.ndarray, noise_floor: float = 1e-3) -> np.ndarray:
    norm_1 = np.linalg.norm(signals_1, ord=2, axis=1)
    norm_2 = np.linalg.norm(signals_2, ord=2, axis=1)

    dot_product = np.abs(np.sum(signals_1 * np.conj(signals_2), axis=-1))
    cos_sim = dot_product / (norm_1 * norm_2 + 1e-8)
    #if the channel only contain noise there would never be similarity because its random variables
    energy_1 = np.mean(np.abs(signals_1)**2, axis=-1)
    energy_2 = np.mean(np.abs(signals_2)**2, axis=-1)
    
    # If both signals are below noise floor, they are functionally identical (silent)
    is_noise = (energy_1 < noise_floor) & (energy_2 < noise_floor)
    return np.where(is_noise, 1.0, cos_sim)