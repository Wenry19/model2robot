
import sys

from torch.utils.data import DataLoader

from model2robot.inference.pytorch_inference import PyTorchInference
from model2robot.inference.tensorrt_inference import TensorRTInference

from model2robot.benchmarking.benchmark import Benchmark
from model2robot.benchmarking.report import generate_report

import model2robot.utils as utils

from models.detectdoor import Detectdoor
from datasets.simple_dataset import SimpleDataset
from datasets.transforms import get_default_transform
from datasets.utils import get_images_paths_and_labels

def main():

    ### GET READY ###
    device = utils.find_device()
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### DATASETS ###

    test_img_paths, test_labels = get_images_paths_and_labels(config["dataset"]["test_path"],
                                                              config["dataset"]["class_names"])

    transform = get_default_transform(input_height=config["model"]["input_height"],
                                      input_width=config["model"]["input_width"])

    test_dataset = SimpleDataset(img_paths=test_img_paths,
                                 labels=test_labels,
                                 transform=transform)

    ### DATALOADER ###
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config["dataloader"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["dataloader"]["num_workers"],
                                 pin_memory=config["dataloader"]["pin_memory"],
                                 drop_last=True)

    ### INFERENCE INSTANCE ###

    if config["inference"]["backend"] == "pytorch":
        model_instance = Detectdoor()
        model_instance.to(device)
        inference = PyTorchInference(model_path=config["model"]["path"],
                                     model_instance=model_instance,
                                     device=device)
    elif config["inference"]["backend"] == "tensorrt":
        inference = TensorRTInference(model_path=config["model"]["path"],
                                      batch_size=config["dataloader"]["batch_size"],
                                      input_height=config["model"]["input_height"],
                                      input_width=config["model"]["input_width"])
    else:
        raise ValueError(f"Unsupported backend: {config['inference']['backend']}")

    ### BENCHMARK ###
    benchmark = Benchmark(inference=inference,
                          warmup_iterations=config["benchmark"]["warmup_iterations"],
                          benchmark_iterations=config["benchmark"]["benchmark_iterations"])
    stats, latencies = benchmark.run(test_dataloader)
    benchmark.close()
    inference.close()

    ### RESULTS ###
    generate_report(config["output_path"], stats, latencies)

if __name__ == "__main__":
    main()
