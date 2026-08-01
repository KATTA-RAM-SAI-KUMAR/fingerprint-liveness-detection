import os
import matplotlib.pyplot as plt


def plot_training(history):

    os.makedirs("outputs", exist_ok=True)

    # ------------------------
    # Loss Curve
    # ------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/loss_curve.png")

    plt.close()

    # ------------------------
    # Accuracy Curve
    # ------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_acc"],
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training vs Validation Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/accuracy_curve.png")

    plt.close()

    print("Training plots saved successfully.")