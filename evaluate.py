
import sys
import os

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

import utils

from datasets.simple_dataset import SimpleDataset
from datasets.utils import get_images_paths_and_labels

from inference.pytorch_inference import PyTorchInference
from inference.tensorrt_inference import TensorRTInference

def predict(inference, dataloader):

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in dataloader:

            outputs = inference.run(images)
            if isinstance(outputs, np.ndarray):
                outputs = torch.from_numpy(outputs)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (all_labels, all_predictions, all_probabilities)

def save_results(output_path, class_names, img_paths, labels, predictions, probabilities):

    # Precision, Recall, F1-score and Macros
    save_metrics(output_path, class_names, labels, predictions)

    # Confusion matrix
    plot_confusion_matrix(output_path, class_names, labels, predictions)

    # Precision - Recall curves
    plot_precision_recall_curves(output_path, class_names, labels, probabilities)

    # Individual predictions
    save_individual_predictions(output_path, class_names, img_paths, labels, predictions, probabilities)

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

    y_true = label_binarize(labels, classes=[0, 1, 2])
    y_score = np.array(probabilities)

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
                            "prediction": predictions,
                            "prob_" + class_names[0]: np.array(probabilities)[:, 0],
                            "prob_" + class_names[1]: np.array(probabilities)[:, 1],
                            "prob_" + class_names[2]: np.array(probabilities)[:, 2]})

    results.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

if __name__ == "__main__":

    ### GET READY ###
    device = utils.find_device()
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### DATASETS ###

    target_path = config["dataset"]["test_path"]
    test_img_paths, test_labels = get_images_paths_and_labels(target_path,
                                                              config["dataset"]["class_names"])

    input_width = config["model"]["input_width"]
    input_height = config["model"]["input_height"]
    transform = transforms.Compose([transforms.Resize((input_height, input_width)),
                                        transforms.ToTensor()])

    test_dataset = SimpleDataset(test_img_paths, test_labels, transform)

    ### DATALOADER ###
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config["evaluation"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["dataloader"]["num_workers"],
                                 pin_memory=config["dataloader"]["pin_memory"])

    ### INFERENCE INSTANCE ###

    model_path = config["model"]["path"]
    model_extension = model_path.split(".")[-1]
    
    if model_extension == "pth": # pytorch
        inference = PyTorchInference(model_path, device)
    elif model_extension == "engine": # tensorrt
        inference = TensorRTInference(model_path,
                                      config["evaluation"]["batch_size"],
                                      input_height,
                                      input_width)
    else:
        print("Model extension not recognised!")
        sys.exit(1)

    ### PREDICT ###
    labels, predictions, probabilities = predict(inference, test_dataloader)

    ### RESULTS ###
    save_results(config["output_path"],
                 config["dataset"]["class_names"],
                 test_img_paths,
                 labels,
                 predictions,
                 probabilities)
