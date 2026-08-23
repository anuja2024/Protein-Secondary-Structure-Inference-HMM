import numpy as np
import bayesflow
import keras

from simulator import AA_TO_IDX

WINDOW = 15
DECISION_THRESHOLD = 0.31


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def encode_sequence(seq: str) -> np.ndarray:
    seq = seq.upper().strip()
    invalid = set(seq) - set(AA_TO_IDX.keys())
    if invalid:
        raise ValueError(f"Invalid amino acid letters: {invalid}")
    return np.array([AA_TO_IDX[c] for c in seq])


def predict_sequence(approximator, seq: str, window: int = WINDOW) -> np.ndarray:
    aa = encode_sequence(seq)
    half = window // 2
    length = len(aa)

    padded = np.pad(aa, (half, half), mode="edge")

    windows = []
    for center in range(half, half + length):
        windows.append(padded[center - half: center + half + 1])

    windows = np.array(windows, dtype="int32")
    estimates = approximator.estimate(conditions={"aa_window": windows})
    logits = estimates["target_probs"]["probs"]["logits"]
    p_alpha = softmax(logits)[:, 1]

    return p_alpha


def main():
    approximator = keras.saving.load_model("../outputs/bayesflow_approximator.keras")

    seq = "FVNQHLCGSHLVEALELVCGERGGFYTPK"
    probs = predict_sequence(approximator, seq, window=WINDOW)

    print(f"{'pos':>4} {'aa':>3} {'P(alpha)':>9} {'state':>6}")
    for i, (aa_char, p) in enumerate(zip(seq, probs)):
        state = "H" if p > DECISION_THRESHOLD else "O"
        print(f"{i+1:4d} {aa_char:>3} {p:9.2f} {state:>6}")


if __name__ == "__main__":
    main()