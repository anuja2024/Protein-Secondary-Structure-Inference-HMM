import numpy as np
import pandas as pd
import streamlit as st
import bayesflow
import keras
from pathlib import Path

from simulator import AA_TO_IDX

MODEL_PATH = Path(__file__).parent.parent / "outputs" / "bayesflow_approximator.keras"
WINDOW = 15


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@st.cache_resource
def load_model():
    return keras.saving.load_model(str(MODEL_PATH))


def encode_sequence(seq: str) -> np.ndarray:
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


st.set_page_config(page_title="Protein Secondary Structure Predictor", layout="centered")
st.title("Protein Secondary Structure Predictor")
st.write("Predicts alpha-helix (H) vs. other (O) per residue using a BayesFlow-trained windowed approximator.")

sequence_input = st.text_area(
    "Enter an amino acid sequence",
    value="FVNQHLCGSHLVEALELVCGERGGFYTPK",
    height=100,
)

if st.button("Predict"):
    sequence = sequence_input.upper().strip()
    valid_letters = set(AA_TO_IDX.keys())
    invalid = set(sequence) - valid_letters

    if invalid:
        st.error(f"Invalid amino acid letters found: {sorted(invalid)}")
    elif len(sequence) == 0:
        st.warning("Please enter a sequence.")
    else:
        model = load_model()
        probs = predict_sequence(model, sequence)
        states = np.where(probs > 0.5, "H", "O")

        st.subheader("Predicted secondary structure")
        st.code("".join(states), language=None)

        st.subheader("Detailed table")
        table_df = pd.DataFrame({
            "position": np.arange(1, len(sequence) + 1),
            "amino_acid": list(sequence),
            "P(alpha)": np.round(probs, 3),
            "state": states,
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)