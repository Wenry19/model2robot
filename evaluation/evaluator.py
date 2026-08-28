
import torch
import numpy as np

class Evaluator:    

    def __init__(self, inference):
        self.inference = inference

    def predict(self, dataloader):

        all_labels = []
        all_predictions = []
        all_probabilities = []

        for images, labels in dataloader:

            self.inference.run(images)
            outputs = self.inference.get_output()

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
            all_probabilities.extend(probabilities.numpy())

        return (np.asarray(all_labels),
                np.asarray(all_predictions),
                np.asarray(all_probabilities))
