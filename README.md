# Protein-Secondary-Structure-Inference-HMM

Academic project on predicting protein secondary structure using a two-state Hidden Markov Model (HMM) and BayesFlow.

Overview

The project models protein secondary structure with two hidden states:

H — alpha-helix

O — other (beta-sheets and coils)

A fixed two-state HMM generates amino-acid sequences. The Forward-Backward algorithm calculates exact posterior probabilities for the hidden states. These simulations are then used to train a BayesFlow neural posterior estimator.

Amino-acid sequence
        ↓
Two-state HMM simulator
        ↓
Forward-Backward posterior
        ↓
BayesFlow neural estimator
        ↓
Predicted P(alpha-helix)

The trained model is evaluated on simulated sequences and applied to real protein data, including human insulin (PDB ID 1A7F).

Objectives

Simulate amino-acid sequences with a two-state HMM.

Calculate exact hidden-state posterior probabilities.

Train a BayesFlow posterior estimator.

Compare BayesFlow predictions with Forward-Backward results.

Evaluate recovery and calibration.

Apply the trained model to real protein sequences.

HMM

The simulator starts in the O state.

Current state

Next H

Next O

H

0.90

0.10

O

0.05

0.95

Amino acids are generated using fixed emission probabilities conditional on the hidden state.

Evaluation

The project includes:

Training convergence

Recovery check

Calibration curve

Accuracy

Precision

Recall

F1-score

Matthews correlation coefficient

MAE

MSE

The recovery check compares BayesFlow posterior estimates with the exact Forward-Backward posterior.

Reported recovery results:

MAE         0.0390
MSE         0.0029
Correlation 0.9733

For real-data classification, a decision threshold of 0.31 is used:

P(alpha-helix) > 0.31  → H
P(alpha-helix) ≤ 0.31  → O

Human Insulin

The trained model is applied to human insulin:

PDB ID: 1A7F

Predicted alpha-helix probabilities are compared with the available secondary-structure annotations for the insulin chains.

Project Structure

protein_hmm_sbi/
├── src/
│   ├── simulator.py
│   ├── windowing.py
│   ├── forward_backward.py
│   ├── bayesflow_model.py
│   ├── sbc.py
│   ├── predict_bayesflow.py
│   └── real_data_validation.py
├── figures/
├── outputs/
├── requirements.txt
└── README.md

The exact files may vary depending on the final project version.

Installation

The project uses Python 3.12.

Windows

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Running

From the src directory, run the relevant scripts.

Example:

python sbc.py

The trained BayesFlow model is loaded from the outputs/ directory by the prediction and diagnostic scripts.

Dataset

Real-data validation uses a cleaned protein secondary-structure dataset derived from Protein Data Bank annotations and obtained through Kaggle.

The project converts the available secondary-structure annotations into the binary states:

H = alpha-helix
O = not alpha-helix

Main Technologies

Python 3.12

NumPy

pandas

SciPy

Matplotlib

scikit-learn

hmmlearn

BayesFlow

TensorFlow / Keras

Biopython

See requirements.txt for dependencies and versions.

Limitations

Secondary structure is simplified to two states.

Beta-sheets and coils are grouped into O.

The estimator uses a fixed local sequence window.

HMM transition and emission probabilities are fixed.

The simulator is based on a simplified generative model.
