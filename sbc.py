import numpy as np
import bayesflow
import keras
import matplotlib.pyplot as plt

from simulator import simulate_sequence
from windowing import extract_windows

WINDOW = 15


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def generate_fresh_windows(n_sequences: int, min_length: int, max_length: int, window: int, seed: int):
    rng = np.random.default_rng(seed)
    all_windows, all_fb_targets, all_true_states = [], [], []

    for _ in range(n_sequences):
        length = rng.integers(max(min_length, window), max_length + 1)
        states, aa = simulate_sequence(length)
        w, t = extract_windows(states, aa, window)
        half = window // 2
        true_states = states[half: length - half]
        all_windows.append(w)
        all_fb_targets.append(t)
        all_true_states.append(true_states)

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_fb_targets, axis=0),
        np.concatenate(all_true_states, axis=0),
    )


def recovery_check(approximator, window: int = WINDOW, n_sequences: int = 200, seed: int = 123):
    windows, fb_targets, _ = generate_fresh_windows(n_sequences, 30, 60, window, seed)
    estimates = approximator.estimate(conditions={"aa_window": windows})
    logits = estimates["target_probs"]["probs"]["logits"]
    nn_probs = softmax(logits)[:, 1]

    mae = np.mean(np.abs(fb_targets - nn_probs))
    mse = np.mean((fb_targets - nn_probs) ** 2)
    corr = np.corrcoef(fb_targets, nn_probs)[0, 1]
    print(f"Mean Absolute Error vs exact Forward-Backward: {mae:.4f}")
    print(f"Mean Squared Error vs exact Forward-Backward:  {mse:.4f}")
    print(f"Correlation with exact Forward-Backward:       {corr:.4f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(fb_targets, nn_probs, s=2, alpha=0.3)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("Exact Forward-Backward posterior")
    plt.ylabel("BayesFlow predicted posterior")
    plt.title("Recovery check")
    plt.tight_layout()
    plt.savefig("../outputs/recovery_check_bayesflow.png", dpi=150)
    print("Saved plot to ../outputs/recovery_check_bayesflow.png")


def calibration_curve(approximator, window: int = WINDOW, n_sequences: int = 200, n_bins: int = 10, seed: int = 456):
    windows, _, true_states = generate_fresh_windows(n_sequences, 30, 60, window, seed)
    estimates = approximator.estimate(conditions={"aa_window": windows})
    logits = estimates["target_probs"]["probs"]["logits"]
    nn_probs = softmax(logits)[:, 1]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, observed_freqs = [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (nn_probs >= lo) & (nn_probs < hi)
        if in_bin.sum() > 0:
            bin_centers.append(nn_probs[in_bin].mean())
            observed_freqs.append(true_states[in_bin].mean())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "r--", label="perfect calibration")
    plt.plot(bin_centers, observed_freqs, "o-", label="BayesFlow")
    plt.xlabel("Predicted P(alpha)")
    plt.ylabel("Observed fraction of true alpha states")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/calibration_curve_bayesflow.png", dpi=150)
    print("Saved plot to ../outputs/calibration_curve_bayesflow.png")


def accuracy_check(approximator, window: int = WINDOW, n_sequences: int = 200, seed: int = 789):
    windows, _, true_states = generate_fresh_windows(n_sequences, 30, 60, window, seed)
    estimates = approximator.estimate(conditions={"aa_window": windows})
    logits = estimates["target_probs"]["probs"]["logits"]
    nn_probs = softmax(logits)[:, 1]

    predicted_states = (nn_probs > 0.5).astype(int)
    accuracy = (predicted_states == true_states).mean()
    print(f"Accuracy vs true hidden states: {accuracy:.2%}")


def main():
    print("Loading model...")
    approximator = keras.saving.load_model("../outputs/bayesflow_approximator.keras")
    print("Model loaded.")

    print("Running recovery check...")
    recovery_check(approximator)

    print("Running accuracy check...")
    accuracy_check(approximator)

    print("Running calibration curve...")
    calibration_curve(approximator)


if __name__ == "__main__":
    main()