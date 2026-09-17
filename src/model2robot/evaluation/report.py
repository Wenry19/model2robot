
import os

import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def generate_report(output_path, class_names, img_paths, labels, predictions, probabilities):

    # Precision, Recall, F1-score and Macros
    save_metrics(output_path, class_names, labels, predictions)

    # Individual predictions
    save_individual_predictions(output_path, class_names, img_paths, labels, predictions, probabilities)

    # Confusion matrix
    plot_confusion_matrix(output_path, class_names, labels, predictions)

    # Precision - Recall curves
    plot_precision_recall_curves(output_path, class_names, labels, probabilities)

def save_metrics(output_path, class_names, labels, predictions):

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels,
                                                                                 predictions,
                                                                                 average="macro")

    with open(os.path.join(output_path, "test_metrics.txt"), "w") as f:

        f.write("Evaluation Results\n")
        f.write("============================\n\n")

        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Per-class metrics:\n")
        f.write("------------------\n")

        for i, class_name in enumerate(class_names):
            f.write(
                f"{class_name}:\n"
                f"  Precision: {precision[i]:.4f}\n"
                f"  Recall:    {recall[i]:.4f}\n"
                f"  F1-score:  {f1[i]:.4f}\n"
                f"  Support:   {support[i]}\n\n"
            )

        f.write("Macro averages:\n")
        f.write("---------------\n")
        f.write(f"Precision: {precision_macro:.4f}\n")
        f.write(f"Recall:    {recall_macro:.4f}\n")
        f.write(f"F1-score:  {f1_macro:.4f}\n")

def plot_confusion_matrix(output_path, class_names, labels, predictions):

    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)

    disp.plot()

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "confusion_matrix.png"),
                dpi=300, bbox_inches="tight")

    plt.close()

def plot_precision_recall_curves(output_path, class_names, labels, probabilities):

    y_true = label_binarize(labels, classes=list(range(len(class_names))))
    y_score = probabilities

    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(class_names):

        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])

        average_precision = average_precision_score(y_true[:, i], y_score[:, i])

        plt.plot(recall, precision,
                 label=f"{class_name} (AP={average_precision:.3f})")


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.grid()

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "precision_recall_curve.png"),
                dpi=300, bbox_inches="tight")

    plt.close()

def save_individual_predictions(output_path, class_names, img_paths, labels, predictions, probabilities):

    results = pd.DataFrame({"image": img_paths,
                            "label": labels,
                            "prediction": predictions})

    for i, class_name in enumerate(class_names):
        results[f"prob_{class_name}"] = probabilities[:, i]

    results.to_csv(os.path.join(output_path, "predictions.csv"), index=False)
