
import numpy as np

AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
N_AMINO_ACIDS = len(AMINO_ACIDS)

STATES = ["other", "alpha"]
OTHER, ALPHA = 0, 1
N_STATES = 2

TRANSITION = np.array([
    [0.95, 0.05],
    [0.10, 0.90],
])

EMISSION_ALPHA = np.array([
    12, 6, 3, 5, 1, 9, 5, 4, 2, 7,
    12, 6, 3, 4, 2, 5, 4, 1, 3, 6
]) / 100.0

EMISSION_OTHER = np.array([
    6, 5, 5, 6, 2, 5, 3, 9, 3, 5,
    8, 6, 2, 4, 6, 7, 6, 1, 4, 7
]) / 100.0

EMISSION = np.stack([EMISSION_OTHER, EMISSION_ALPHA])


def simulate_sequence(length: int):
    rng = np.random.default_rng(42)
    
    states = np.zeros(length, dtype=int)
    amino_acids = np.zeros(length, dtype=int)

    states[0] = OTHER
    amino_acids[0] = rng.choice(N_AMINO_ACIDS, p=EMISSION[states[0]])

    for t in range(1, length):
        states[t] = rng.choice(N_STATES, p=TRANSITION[states[t - 1]])
        amino_acids[t] = rng.choice(N_AMINO_ACIDS, p=EMISSION[states[t]])

    return states, amino_acids


def decode_amino_acids(amino_acid_indices: np.ndarray) -> str:
    return "".join(AMINO_ACIDS[i] for i in amino_acid_indices)


if __name__ == "__main__":
    states, aa = simulate_sequence(30)
    print("Hidden states :", states)
    print("Amino acids   :", decode_amino_acids(aa))
    print(f"Fraction alpha: {states.mean():.2f}")