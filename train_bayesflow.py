import bayesflow as bf
import matplotlib.pyplot as plt

from windowing import build_window_dataset
from bayesflow_model import build_approximator


def main():
    N_TRAIN, N_VAL = 800, 200
    MIN_LEN, MAX_LEN = 3, 400
    WINDOW = 15
    BATCH_SIZE = 64
    N_EPOCHS = 20
    SEED = 42

    print("Generating training windows...")
    train_windows, train_targets = build_window_dataset(
        N_TRAIN, MIN_LEN, MAX_LEN, WINDOW, seed=SEED
    )
    print(f"  {train_windows.shape[0]} training windows")

    print("Generating validation windows...")
    val_windows, val_targets = build_window_dataset(
        N_VAL, MIN_LEN, MAX_LEN, WINDOW, seed=SEED + 1
    )
    print(f"  {val_windows.shape[0]} validation windows")

    approximator = build_approximator()
    approximator.compile(optimizer="adam")

    train_data = {"aa_window": train_windows, "target_probs": train_targets}
    val_data = {"aa_window": val_windows, "target_probs": val_targets}

    train_dataset = bf.datasets.OfflineDataset(
        data=train_data, batch_size=BATCH_SIZE, adapter=approximator.adapter
    )
    val_dataset = bf.datasets.OfflineDataset(
        data=val_data, batch_size=BATCH_SIZE, adapter=approximator.adapter, shuffle=False
    )

    history = approximator.fit(dataset=train_dataset, validation_data=val_dataset, epochs=N_EPOCHS)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/training_curve.png", dpi=150)
    print("Saved plot to ../outputs/training_curve.png")

    approximator.save("../outputs/bayesflow_approximator.keras")
    print("Saved trained approximator to ../outputs/bayesflow_approximator.keras")


if __name__ == "__main__":
    main()