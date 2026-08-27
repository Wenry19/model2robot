
import sys

from torchvision import transforms
from torch.utils.data import DataLoader

import utils

from datasets.utils import get_images_paths_and_labels

from inference.pytorch_inference import PyTorchInference
from inference.tensorrt_inference import TensorRTInference

from evaluation.evaluator import Evaluator
from evaluation.report import generate_report

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

    transform_config = config["dataset"]["transform"]
    transform = utils.import_function(transform_config["module"],
                                      transform_config["function"],
                                      **transform_config.get("args", {}))

    test_dataset = utils.instantiate_class(config["dataset"]["module"],
                                           config["dataset"]["class"],
                                           img_paths=test_img_paths,
                                           labels=test_labels,
                                           transform=transform,
                                           **config["dataset"].get("args", {}))

    ### DATALOADER ###
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config["evaluation"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["dataloader"]["num_workers"],
                                 pin_memory=config["dataloader"]["pin_memory"])

    ### INFERENCE INSTANCE ###

    model_path = config["model"]["path"]
    model_module = config["model"]["module"]
    model_class = config["model"]["class"]
    model_extension = model_path.split(".")[-1]
    
    if config["model"]["type"] == "pytorch":
        inference = PyTorchInference(model_path, model_module, model_class, device)
    elif config["model"]["type"] == "tensorrt":
        inference = TensorRTInference(model_path,
                                      config["evaluation"]["batch_size"],
                                      input_height,
                                      input_width)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    ### PREDICT ###
    evaluator = Evaluator(inference)
    labels, predictions, probabilities = evaluator.predict(test_dataloader)

    ### RESULTS ###
    generate_report(config["output_path"],
                    config["dataset"]["class_names"],
                    test_img_paths,
                    labels,
                    predictions,
                    probabilities)

    inference.close()
