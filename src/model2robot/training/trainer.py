
import os
import random
import numpy as np
import torch
from datetime import datetime

class Trainer:

    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 device,
                 loss_func,
                 optimizer,
                 output_path):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.output_path = output_path

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.timestamps = []

        self.best_val_loss = None

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        
        for images, labels in self.train_dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # PyTorch accumulates gradients by default
            self.optimizer.zero_grad() # re-initialize gradients

            # forward pass
            outputs = self.model(images) # output shape: (batch_size, num_classes)

            # compute loss
            loss = self.loss_func(outputs, labels)

            # compute gradients
            loss.backward()
            # to see gradients: parameter.grad

            # update weights
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_dataloader)

    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.loss_func(outputs, labels)

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

    def save_checkpoint(self, val_loss, val_acc, epoch):

        if self.best_val_loss is None or val_loss < self.best_val_loss:

            self.best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                os.path.join(self.output_path, "best_model.pth")
            )

            print(f"Saved new best model (val_loss={val_loss:.3f})")

    def train(self, epochs):

        for epoch in range(epochs):
            
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"{timestamp} | "
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.3f} "
                    f"val_loss={val_loss:.3f} "
                    f"val_acc={val_acc:.2f}")
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.timestamps.append(timestamp)
            
            self.save_checkpoint(val_loss, val_acc, epoch)
