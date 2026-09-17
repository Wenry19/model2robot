
import sys
import yaml
import torch
import os

def find_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    return device

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("Output path already exists:", output_path)
        sys.exit(1)

def read_config_file(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_config(config):
    output_config_path = os.path.join(config["output_path"],
                                      "config.yaml")
    with open(output_config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
