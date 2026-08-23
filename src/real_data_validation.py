import numpy as np
import pandas as pd
import bayesflow
import keras

from simulator import AA_TO_IDX
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

CSV_PATH = "C:/protein_hmm_sbi/data/protein_cleaned.csv"
MODEL_PATH = "C:/protein_hmm_sbi/outputs/bayesflow_approximator.keras"
WINDOW = 15
DECISION_THRESHOLD = 0.31

VALID_AMINO_ACIDS = set(AA_TO_IDX.keys())


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sst3_to_binary(sst3: str) -> np.ndarray:
    return np.array([1 if c == "H" else 0 for c in sst3])


def is_encodable(seq: str) -> bool:
    return all(c in VALID_AMINO_ACIDS for c in seq)


def encode_sequence(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX[c] for c in seq])


def build_padded_windows(seq: str, window: int = WINDOW) -> np.ndarray:
    aa = encode_sequence(seq)
    half = window // 2
    padded = np.pad(aa, (half, half), mode="edge")
    return np.array([padded[i:i + window] for i in range(len(aa))], dtype="int32")


def evaluate_sample(approximator, df: pd.DataFrame, n_samples: int, max_length: int, seed: int = 0):
    candidates = df[(df["has_nonstd_aa"] == False) & (df["len"] <= max_length)]
    candidates = candidates[candidates["seq"].apply(is_encodable)]
    sample = candidates.sample(n=min(n_samples, len(candidates)), random_state=seed)

    all_windows, all_true = [], []
    for _, row in sample.iterrows():
        all_windows.append(build_padded_windows(row["seq"]))
        all_true.append(sst3_to_binary(row["sst3"]))

    windows = np.concatenate(all_windows, axis=0)
    true_flat = np.concatenate(all_true, axis=0)

    estimates = approximator.estimate(conditions={"aa_window": windows})
    logits = estimates["target_probs"]["probs"]["logits"]
    preds_flat = softmax(logits)[:, 1]

    hard_preds = (preds_flat > DECISION_THRESHOLD).astype(int)
    accuracy = (hard_preds == true_flat).mean()
    mae = np.mean(np.abs(preds_flat - true_flat))
    mse = np.mean((preds_flat - true_flat) ** 2)
    precision = precision_score(true_flat, hard_preds)
    recall = recall_score(true_flat, hard_preds)
    f1 = f1_score(true_flat, hard_preds)
    mcc = matthews_corrcoef(true_flat, hard_preds)

    print(f"Evaluated on {len(sample)} real sequences ({len(preds_flat)} residues)")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"MSE:       {mse:.4f}")


def evaluate_insulin(approximator, df: pd.DataFrame):
    insulin = df[df["pdb_id"] == "1A7F"]

    for _, row in insulin.iterrows():
        seq, sst3, chain = row["seq"], row["sst3"], row["chain_code"]
        true_states = sst3_to_binary(sst3)
        windows = build_padded_windows(seq)

        estimates = approximator.estimate(conditions={"aa_window": windows})
        logits = estimates["target_probs"]["probs"]["logits"]
        probs = softmax(logits)[:, 1]
        hard_preds = (probs > DECISION_THRESHOLD).astype(int)
        accuracy = (hard_preds == true_states).mean()

        print(f"\n=== Insulin chain {chain} ({len(seq)} residues) ===")
        print(f"{'pos':>4} {'aa':>3} {'true':>5} {'pred_prob':>10} {'pred_state':>11}")
        for i, (aa_char, t, p) in enumerate(zip(seq, true_states, probs)):
            print(f"{i+1:4d} {aa_char:>3} {t:5d} {p:10.2f} {int(p > DECISION_THRESHOLD):11d}")
        print(f"Accuracy: {accuracy:.2%}")


def main():
    df = pd.read_csv(CSV_PATH)
    approximator = keras.saving.load_model(MODEL_PATH)

    print("=== Aggregate evaluation on real sequences ===")
    evaluate_sample(approximator, df, n_samples=2000, max_length=300)

    print("\n=== Worked example: human insulin ===")
    evaluate_insulin(approximator, df)


if __name__ == "__main__":
    main()