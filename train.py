
import sys
import os

import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from models.detectdoor import Detectdoor
from datasets.simple_dataset import SimpleDataset
from datasets.utils import get_images_paths_and_labels

LOSS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "BCELoss": nn.BCELoss,
}

OPTIMIZER = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, device, loss_func, optimizer):

    model.train()

    running_loss = 0.0
    
    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # PyTorch accumulates gradients by default
        optimizer.zero_grad() # re-initialize gradients

        # forward pass
        outputs = model(images) # output shape: (batch_size, num_classes)

        # compute loss
        loss = loss_func(outputs, labels)

        # compute gradients
        loss.backward()
        # to see gradients: parameter.grad

        # update weights
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate(model, dataloader, device, loss_func):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_func(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Predictions
            predictions = torch.argmax(outputs, dim=1)

            # Accuracy
            total_correct += (predictions == labels).sum().item()

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy

def save_checkpoint(output_path, best_val_loss, val_loss, val_acc, epoch, model, optimizer):

    if best_val_loss is None or val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            os.path.join(output_path, "best_model.pth")
        )

        print(f"Saved new best model (val_loss={val_loss:.3f})")

    return best_val_loss

def save_train_metrics(output_path, timestamps, train_losses, val_losses, val_accuracies):

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

    # PLOTS
    
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

if __name__ == "__main__":

    ### GET READY ###
    device = utils.find_device()
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)
    set_seed(config["seed"])

    ### MODEL ###
    model = Detectdoor()
    model.to(device)

    ### LOSS FUNCTION ###
    loss_func = LOSS[config["training"]["loss"]]()
    # expects (batch_size, num_classes) which are the predictions (one-hot vectors)
    # and (batch_size) which are the labels (integers)

    ### OPTIMIZER ###
    optimizer = OPTIMIZER[config["training"]["optimizer"]](model.parameters(),
                                                           lr=config["training"]["learning_rate"])

    ### DATASETS ###

    target_path = config["dataset"]["train_path"]
    train_img_paths, train_labels = get_images_paths_and_labels(target_path,
                                                                config["dataset"]["class_names"])

    target_path = config["dataset"]["val_path"]
    val_img_paths, val_labels = get_images_paths_and_labels(target_path,
                                                            config["dataset"]["class_names"])

    input_width = config["model"]["input_width"]
    input_height = config["model"]["input_height"]
    transform = transforms.Compose([transforms.Resize((input_height, input_width)),
                                    transforms.ToTensor()])

    train_dataset = SimpleDataset(train_img_paths, train_labels, transform)
    val_dataset = SimpleDataset(val_img_paths, val_labels, transform)
    # image, label = train_dataset[0]

    ### DATALOADER ###

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["training"]["batch_size"],
                                  shuffle=config["dataloader"]["shuffle"],
                                  num_workers=config["dataloader"]["num_workers"],
                                  pin_memory=config["dataloader"]["pin_memory"],
                                  drop_last=config["dataloader"]["drop_last"])

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["training"]["batch_size"],
                                shuffle=False,
                                num_workers=config["dataloader"]["num_workers"],
                                pin_memory=config["dataloader"]["pin_memory"])

    ### TRAINING LOOP ###

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = None
    timestamps = []

    for epoch in range(config["training"]["epochs"]):
        
        train_loss = train_one_epoch(model, train_dataloader, device, loss_func, optimizer)
        val_loss, val_acc = validate(model, val_dataloader, device, loss_func)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{timestamp} | "
              f"Epoch {epoch}: "
              f"train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} "
              f"val_acc={val_acc:.2f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        timestamps.append(timestamp)
        
        best_val_loss = save_checkpoint(config["output_path"],
                                        best_val_loss,
                                        val_loss,
                                        val_acc,
                                        epoch,
                                        model,
                                        optimizer)

    save_train_metrics(config["output_path"],
                       timestamps,
                       train_losses,
                       val_losses,
                       val_accuracies)
