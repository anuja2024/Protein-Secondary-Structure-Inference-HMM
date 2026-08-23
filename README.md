# Simulation-Based Inference of Protein Secondary Structure

Academic project on predicting protein secondary structure using a two-state Hidden Markov Model (HMM) and BayesFlow.

## Project Overview

The project models protein secondary structure with two hidden states:

- `H` — alpha-helix
- `O` — other (beta-sheets and coils)

A fixed two-state HMM generates amino-acid sequences. The Forward-Backward algorithm calculates exact posterior probabilities for the hidden states. These simulations are then used to train a BayesFlow neural posterior estimator.

```text
Amino-acid sequence
        |
        v
Two-state HMM simulator
        |
        v
Forward-Backward posterior
        |
        v
BayesFlow neural estimator
        |
        v
Predicted P(alpha-helix)
```

The trained model is evaluated on simulated sequences and applied to real protein data, including human insulin (PDB ID `1A7F`).

## Objectives

1. Simulate amino-acid sequences with a two-state HMM.
2. Calculate exact hidden-state posterior probabilities.
3. Train a BayesFlow posterior estimator.
4. Compare BayesFlow predictions with Forward-Backward results.
5. Evaluate recovery and calibration.
6. Apply the trained model to real protein sequences.

## Hidden Markov Model

The simulator starts in the `O` state.

| Current state | Next H | Next O |
|---|---:|---:|
| H | 0.90 | 0.10 |
| O | 0.05 | 0.95 |

Amino acids are generated using fixed emission probabilities conditional on the hidden state.

## Evaluation

The project includes:

- Training convergence
- Recovery check
- Calibration curve
- Accuracy
- Precision
- Recall
- F1-score
- Matthews correlation coefficient
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

The recovery check compares BayesFlow posterior estimates with the exact Forward-Backward posterior.

Reported recovery results:

```text
MAE          0.0390
MSE          0.0029
Correlation  0.9733
```

For real-data classification, a decision threshold of **0.31** is used:

```text
P(alpha-helix) > 0.31   -> H
P(alpha-helix) <= 0.31  -> O
```

## Human Insulin

The trained model is applied to human insulin:

```text
PDB ID: 1A7F
```

Predicted alpha-helix probabilities are compared with the available secondary-structure annotations for the insulin chains.

## Project Structure

```text
protein_hmm_sbi/
|
+-- src/
|   +-- simulator.py
|   +-- windowing.py
|   +-- forward_backward.py
|   +-- bayesflow_model.py
|   +-- sbc.py
|   +-- predict_bayesflow.py
|   +-- real_data_validation.py
|
+-- figures/
+-- outputs/
+-- requirements.txt
+-- README.md
```

The exact files may vary depending on the final project version.

## Installation

The project uses **Python 3.12**.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

From the `src` directory, run the relevant scripts.

Example:

```powershell
python sbc.py
```

The trained BayesFlow model is loaded from the `outputs/` directory by the prediction and diagnostic scripts.

## Dataset

Real-data validation uses a cleaned protein secondary-structure dataset derived from Protein Data Bank annotations and obtained through Kaggle.

The available secondary-structure annotations are converted into the binary states:

```text
H = alpha-helix
O = not alpha-helix
```

## Main Technologies

- Python 3.12
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- hmmlearn
- BayesFlow
- TensorFlow / Keras
- Biopython

See `requirements.txt` for the project dependencies and versions.

## Limitations

- Secondary structure is simplified to two states.
- Beta-sheets and coils are grouped into `O`.
- The estimator uses a fixed local sequence window.
- HMM transition and emission probabilities are fixed.
- The simulator is based on a simplified generative model.

## Authors

- Anuja Patade
- Ashmi Desai
- Binal Dave
- Svetlana Färber

**TU Dortmund University — Simulation-Based Inference**
