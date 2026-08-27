
import sys
import json
import torch
import os
import importlib

def find_device():
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    return device

def read_config_file(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("Output path already exists:", output_path)
        sys.exit(1)

def save_config(config):
    output_config_path = os.path.join(config["output_path"],
                                      "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=4)

def instantiate_class(module_name, class_name, *args, **kwargs):

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)

def import_function(module_name, function_name, **kwargs):

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    return function(**kwargs)