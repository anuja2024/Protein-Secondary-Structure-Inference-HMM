# forward_backward.py
import numpy as np
from hmmlearn import hmm

from simulator import TRANSITION, EMISSION, ALPHA, N_AMINO_ACIDS, N_STATES


def build_known_hmm() -> hmm.CategoricalHMM:
    model = hmm.CategoricalHMM(n_components=N_STATES, n_features=N_AMINO_ACIDS, init_params="")
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = TRANSITION
    model.emissionprob_ = EMISSION
    return model


def forward_backward_posterior(amino_acid_seq: np.ndarray) -> np.ndarray:
    model = build_known_hmm()
    X = amino_acid_seq.reshape(-1, 1)
    posteriors = model.predict_proba(X)
    return posteriors[:, ALPHA]


def batch_forward_backward(amino_acid_batch: np.ndarray) -> np.ndarray:
    n, length = amino_acid_batch.shape
    out = np.zeros((n, length))
    for i in range(n):
        out[i] = forward_backward_posterior(amino_acid_batch[i])
    return out


if __name__ == "__main__":
    from simulator import simulate_sequence, decode_amino_acids

    states, aa = simulate_sequence(30)
    posterior = forward_backward_posterior(aa)

    print("Amino acids   :", decode_amino_acids(aa))
    print("True states   :", states)
    print("Posterior P(a):", np.round(posterior, 2))