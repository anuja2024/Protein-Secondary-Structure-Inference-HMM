
import numpy as np

from simulator import simulate_sequence
from forward_backward import forward_backward_posterior


def extract_windows(states: np.ndarray, amino_acids: np.ndarray, window: int):
    half = window // 2
    length = len(amino_acids)
    posterior = forward_backward_posterior(amino_acids)

    windows, targets = [], []
    for center in range(half, length - half):
        windows.append(amino_acids[center - half: center + half + 1])
        targets.append(posterior[center])

    return np.array(windows, dtype="int32"), np.array(targets, dtype="float32")


def build_window_dataset(n_sequences: int, min_length: int, max_length: int,
                          window: int, seed: int = None):
    rng = np.random.default_rng(seed)
    all_windows, all_targets = [], []

    for _ in range(n_sequences):
        length = rng.integers(max(min_length, window), max_length + 1)
        states, aa = simulate_sequence(length)
        w, t = extract_windows(states, aa, window)
        all_windows.append(w)
        all_targets.append(t)

    windows = np.concatenate(all_windows, axis=0)
    center_p = np.concatenate(all_targets, axis=0)
    target_probs = np.stack([1 - center_p, center_p], axis=1).astype("float32")

    return windows, target_probs


if __name__ == "__main__":
    windows, target_probs = build_window_dataset(
        n_sequences=20, min_length=30, max_length=60, window=15, seed=0
    )
    print("windows shape     :", windows.shape)
    print("target_probs shape:", target_probs.shape)