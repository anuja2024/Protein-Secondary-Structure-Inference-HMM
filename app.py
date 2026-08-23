import os

# Set the Keras backend BEFORE importing Keras or BayesFlow.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import streamlit as st
import bayesflow
import keras
from pathlib import Path

from src.simulator import AA_TO_IDX

MODEL_PATH = Path(__file__).parent / "outputs" / "bayesflow_approximator.keras"

WINDOW = 15
THRESHOLD = 0.31

def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@st.cache_resource
def load_model():
    """Load the trained BayesFlow approximator."""
    return keras.saving.load_model(str(MODEL_PATH))


def encode_sequence(seq: str) -> np.ndarray:
    """Convert amino-acid letters to integer indices."""
    return np.array([AA_TO_IDX[c] for c in seq])


def predict_sequence(
    approximator,
    seq: str,
    window: int = WINDOW
) -> np.ndarray:
    """Predict P(alpha-helix) for every residue."""

    aa = encode_sequence(seq)

    half = window // 2
    length = len(aa)

    padded = np.pad(
        aa,
        (half, half),
        mode="edge"
    )

    windows = []

    for center in range(half, half + length):
        windows.append(
            padded[
                center - half:
                center + half + 1
            ]
        )

    windows = np.array(
        windows,
        dtype="int32"
    )


    estimates = approximator.estimate(
        conditions={
            "aa_window": windows
        }
    )

    logits = estimates[
        "target_probs"
    ][
        "probs"
    ][
        "logits"
    ]

    p_alpha = softmax(logits)[:, 1]

    return p_alpha


st.set_page_config(
    page_title="Protein Secondary Structure Predictor",
    page_icon="🧬",
    layout="centered"
)

st.title("Protein Secondary Structure Predictor")

st.write(
    "Predicts alpha-helix (H) vs. other (O) per residue "
    "using a BayesFlow-trained windowed approximator."
)


sequence_input = st.text_area(
    "Enter an amino acid sequence",
    value="FVNQHLCGSHLVEALELVCGERGGFYTPK",
    height=100
)


if st.button("Predict"):

    sequence = sequence_input.upper().strip()

    valid_letters = set(AA_TO_IDX.keys())
    invalid = set(sequence) - valid_letters

    if invalid:

        st.error(
            f"Invalid amino acid letters found: "
            f"{sorted(invalid)}"
        )

    elif len(sequence) == 0:

        st.warning(
            "Please enter a sequence."
        )

    else:

        try:

            model = load_model()

            probs = predict_sequence(
                model,
                sequence
            )

        
            states = np.where(
                probs > THRESHOLD,
                "H",
                "O"
            )

        

            st.subheader(
                "Predicted secondary structure"
            )

            st.code(
                "".join(states),
                language=None
            )

        

            st.subheader(
                "Detailed table"
            )

            table_df = pd.DataFrame({

                "position":
                    np.arange(
                        1,
                        len(sequence) + 1
                    ),

                "amino_acid":
                    list(sequence),

                "P(alpha)":
                    np.round(
                        probs,
                        3
                    ),

                "state":
                    states
            })

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True
            )

        
            st.caption(
                f"Window size: {WINDOW} residues | "
                f"Decision threshold: {THRESHOLD}"
            )

        except Exception as e:

            st.error(
                "An error occurred while loading the "
                "model or generating predictions."
            )

            st.exception(e)
