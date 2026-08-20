
import sys
import os
import json

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import pandas as pd

from datasets.simple_dataset import SimpleDataset
from datasets.utils import get_images_paths_and_labels

from models.detectdoor import Detectdoor

def save_test_config(config):
    output_test_config_path = os.path.join(config["results"]["output_path"],
                                           "test_config.json")
    with open(output_test_config_path, "w") as f:
        json.dump(config, f, indent=4)

def predict(model, dataloader, device):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (all_labels, all_predictions, all_probabilities)

def get_results(config, labels, predictions, probabilities):

    # Confusion matrix

    plot_confusion_matrix(config, labels, predictions)

    # Precision, Recall, F1-score and Macros

    save_metrics(config, labels, predictions)

    # Precision - Recall curves

    plot_precision_recall_curves(config, labels, probabilities)

def plot_confusion_matrix(config, labels, predictions):

    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=config["dataset"]["class_names"])

    disp.plot()

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(os.path.join(config["results"]["output_path"], "confusion_matrix.png"),
                dpi=300, bbox_inches="tight")

    plt.close()

def plot_precision_recall_curves(config, labels, probabilities):

    y_true = label_binarize(labels, classes=[0, 1, 2])
    y_score = np.array(probabilities)

    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(config["dataset"]["class_names"]):

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

    plt.savefig(os.path.join(config["results"]["output_path"], "precision_recall_curve.png"),
                dpi=300, bbox_inches="tight")

    plt.close()

def save_metrics(config, labels, predictions):

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels,
                                                                                 predictions,
                                                                                 average="macro")

    with open(os.path.join(config["results"]["output_path"], "test_metrics.txt"), "w") as f:

        f.write("Evaluation Results\n")
        f.write("============================\n\n")

        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Per-class metrics:\n")
        f.write("------------------\n")

        for i, class_name in enumerate(config["dataset"]["class_names"]):
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

if __name__ == "__main__":

    ### FIND DEVICE ###

    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ### READ CONFIG FILE ###

    config_path = sys.argv[1]

    with open(config_path, "r") as file:
        config = json.load(file)

    ### PREPARE OUTPUT DIRECTORY ###
    if not os.path.exists(config["results"]["output_path"]):
        os.makedirs(config["results"]["output_path"])
    else:
        print("Output path already exists:", config["results"]["output_path"])
        sys.exit(1)

    ### SAVE READ CONFIG FOR REPRODUCIBILITY ###
    save_test_config(config)

    ### DATASETS ###

    target_path = config["dataset"]["test_path"]
    test_img_paths, test_labels = get_images_paths_and_labels(target_path, config["dataset"]["class_names"])

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    test_dataset = SimpleDataset(test_img_paths, test_labels, transform)

    ### DATALOADER ###

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config["evaluation"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["evaluation"]["num_workers"],
                                 pin_memory=config["evaluation"]["pin_memory"])

    ### MODEL ###

    checkpoint = torch.load(config["model"]["checkpoint"],
                            map_location=device,
                            weights_only=True)

    model = Detectdoor().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    ### PREDICT ###

    labels, predictions, probabilities = predict(model, test_dataloader, device)

    ### RESULTS ###

    get_results(config, labels, predictions, probabilities)

    ### SAVE INDIVIDUAL PREDICTIONS ###

    results = pd.DataFrame({"image": test_img_paths,
                            "label": labels,
                            "prediction": predictions,
                            "prob_" + config["dataset"]["class_names"][0]: np.array(probabilities)[:, 0],
                            "prob_" + config["dataset"]["class_names"][1]: np.array(probabilities)[:, 1],
                            "prob_" + config["dataset"]["class_names"][2]: np.array(probabilities)[:, 2]})

    results.to_csv(os.path.join(config["results"]["output_path"], "predictions.csv"), index=False)
