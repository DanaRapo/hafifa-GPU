import numpy as np

def energy_similarity(signals_1: np.ndarray, signals_2: np.ndarray) -> np.ndarray:
    """
    Calculates the similarity between the energy profiles (|x|^2) of two signals.
    Matches the style of your cosine_similarity function.
    """
    p1 = np.mean(np.abs(signals_1)**2, axis=1)
    p2 = np.mean(np.abs(signals_2)**2, axis=1)
    
    # Return the ratio. We want this to be very close to 1.0
    # Use p1/p2 or p2/p1, adding epsilon to avoid div by zero
    return p1 / (p2 + 1e-12)