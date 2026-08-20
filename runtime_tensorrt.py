
import sys
import os
import json

import tensorrt as trt
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.utils import get_images_paths_and_labels
from datasets.simple_dataset import SimpleDataset

from tensorrt_inference import TensorRTInference

def predict(config, engine, dataloader):

    inference = TensorRTInference(config, engine)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in dataloader:

        images = np.ascontiguousarray(images, dtype=np.float32)

        outputs = inference.run(images)
        outputs = torch.from_numpy(outputs)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    inference.close()

    return (all_labels, all_predictions, all_probabilities)

def save_runtime_config(config):
    output_runtime_config_path = os.path.join(config["results"]["output_path"],
                                              "runtime_tensorrt_config.json")
    with open(output_runtime_config_path, "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":

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
    save_runtime_config(config)

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

    ### ENGINE ###

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(config["model"]["engine"], "rb") as f:
        model_data = f.read()
    engine = runtime.deserialize_cuda_engine(model_data)
    
    ### PREDICT ###

    labels, predictions, probabilities = predict(config, engine, test_dataloader)

    ### RESULTS ###

    # TODO: could be reused from evaluate.py
    # TODO: add inference time in the results, and GPU usage, stats, etc.

    ### SAVE INDIVIDUAL PREDICTIONS ###

    # TODO: could be reused from evaluate.py
