
import sys
import os
import json

import torch
import onnx

from models.detectdoor import Detectdoor

def save_export_config(config):
    output_export_config_path = os.path.join(config["export"]["output_path"],
                                             "export_to_onnx_config.json")
    with open(output_export_config_path, "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":

    ### READ CONFIG FILE ###
    
    config_path = sys.argv[1]

    with open(config_path, "r") as file:
        config = json.load(file)

    ### PREPARE OUTPUT DIRECTORY ###
    if not os.path.exists(config["export"]["output_path"]):
        os.makedirs(config["export"]["output_path"])
    else:
        print("Output path already exists:", config["export"]["output_path"])
        sys.exit(1)

    ### SAVE READ CONFIG FOR REPRODUCIBILITY ###
    save_export_config(config)

    ### DUMMY DATA ###
    # We need a batch of data to save our ONNX file from PyTorch. We will use a dummy batch.
    # Dummy input used by the ONNX exporter to trace the model's computation graph.
    # Values are irrelevant, only the input shape must match the model's expected input.
    dummy_input=torch.randn(config["export"]["batch_size"],
                            3,
                            config["model"]["input_height"],
                            config["model"]["input_width"])

    ### LOAD PYTORCH MODEL TO BE EXPORTED ###
    checkpoint = torch.load(config["model"]["checkpoint"],
                            map_location="cpu",
                            weights_only=True)
    model = Detectdoor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # IMPORTANT!

    ### EXPORT TO ONNX ###
    onnx_file_name = os.path.basename(config["model"]["checkpoint"]).split(".")[0] + ".onnx"
    onnx_output_path = os.path.join(config["export"]["output_path"], onnx_file_name)
    torch.onnx.export(model,
                      dummy_input,
                      onnx_output_path,
                      verbose=False)

    ### CHECK ONNX FILE ###
    
    try:
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model successfully exported and validated")

    except onnx.checker.ValidationError as e:
        print("ONNX model validation failed:")
        print(e)
