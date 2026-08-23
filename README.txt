SIMULATION-BASED INFERENCE OF PROTEIN SECONDARY STRUCTURE
==============================================================

Academic project on predicting protein secondary structure using a
two-state Hidden Markov Model (HMM) and BayesFlow.


PROJECT OVERVIEW
----------------

The project models protein secondary structure with two hidden states:

H = alpha-helix
O = other (beta-sheets and coils)

A fixed two-state HMM generates amino-acid sequences. The Forward-Backward
algorithm calculates exact posterior probabilities for the hidden states.
These simulations are then used to train a BayesFlow neural posterior
estimator.

Workflow:

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

The trained model is evaluated on simulated sequences and applied to real
protein data, including human insulin (PDB ID 1A7F).


OBJECTIVES
----------

1. Simulate amino-acid sequences with a two-state HMM.
2. Calculate exact hidden-state posterior probabilities.
3. Train a BayesFlow posterior estimator.
4. Compare BayesFlow predictions with Forward-Backward results.
5. Evaluate recovery and calibration.
6. Apply the trained model to real protein sequences.


HIDDEN MARKOV MODEL
-------------------

The simulator starts in the O state.

    Current state     Next H     Next O
    ------------------------------------
    H                    0.90       0.10
    O                    0.05       0.95

Amino acids are generated using fixed emission probabilities conditional
on the hidden state.


EVALUATION
----------

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

The recovery check compares BayesFlow posterior estimates with the exact
Forward-Backward posterior.

Reported recovery results:

    MAE          0.0390
    MSE          0.0029
    Correlation  0.9733

For real-data classification, a decision threshold of 0.31 is used:

    P(alpha-helix) > 0.31  -> H
    P(alpha-helix) <= 0.31 -> O


HUMAN INSULIN
-------------

The trained model is applied to human insulin:

    PDB ID: 1A7F

Predicted alpha-helix probabilities are compared with the available
secondary-structure annotations for the insulin chains.


PROJECT STRUCTURE
-----------------

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


INSTALLATION
------------

The project uses Python 3.12.

On Windows:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt


RUNNING THE PROJECT
-------------------

From the src directory, run the relevant scripts.

Example:

    python sbc.py

The trained BayesFlow model is loaded from the outputs directory by the
prediction and diagnostic scripts.


DATASET
-------

Real-data validation uses a cleaned protein secondary-structure dataset
derived from Protein Data Bank annotations and obtained through Kaggle.

The available secondary-structure annotations are converted into the
binary states:

    H = alpha-helix
    O = not alpha-helix


MAIN TECHNOLOGIES
-----------------

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

See requirements.txt for the project dependencies and versions.


LIMITATIONS
-----------

- Secondary structure is simplified to two states.
- Beta-sheets and coils are grouped into O.
- The estimator uses a fixed local sequence window.
- HMM transition and emission probabilities are fixed.
- The simulator is based on a simplified generative model.


AUTHORS
-------

Anuja Patade
Ashmi Desai
Binal Dave
Svetlana Färber

TU Dortmund University
Simulation-Based Inference
