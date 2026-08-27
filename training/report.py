
import os
import csv
import matplotlib.pyplot as plt

def generate_report(output_path, timestamps, train_losses, val_losses, val_accuracies):

    # Training general metrics
    save_metrics(output_path, timestamps, train_losses, val_losses, val_accuracies)

    # Loss curve
    plot_loss(output_path, train_losses, val_losses)

    # Accuracy curve
    plot_accuracy(output_path, val_accuracies)

def save_metrics(output_path, timestamps, train_losses, val_losses, val_accuracies):

    output_train_metrics_path = os.path.join(output_path,
                                             "train_metrics.csv")

    with open(output_train_metrics_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["timestamp",
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_accuracy"])

        for epoch, timestamp, train_loss, val_loss, val_acc in zip(range(len(timestamps)),
                                                                        timestamps,
                                                                        train_losses,
                                                                        val_losses,
                                                                        val_accuracies):
            writer.writerow([timestamp,
                            epoch,
                            train_loss,
                            val_loss,
                            val_acc])

def plot_loss(output_path, train_losses, val_losses):

    epochs = range(len(train_losses))

    # Loss curve

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(output_path, "loss_curve.png"), bbox_inches="tight")
    plt.close()

def plot_accuracy(output_path, val_accuracies):

    epochs = range(len(val_accuracies))

    # Accuracy curve

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(output_path, "accuracy_curve.png"), bbox_inches="tight")
    plt.close()
