
import sys
import os

import torch
import onnx

import utils

if __name__ == "__main__":

    ### GET READY ###
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### DUMMY DATA ###
    # We need a batch of data to save our ONNX file from PyTorch. We will use a dummy batch.
    # Dummy input used by the ONNX exporter to trace the model's computation graph.
    # Values are irrelevant, only the input shape must match the model's expected input.
    dummy_input=torch.randn(config["export"]["batch_size"],
                            3,
                            config["model"]["input_height"],
                            config["model"]["input_width"])

    ### LOAD PYTORCH MODEL TO BE EXPORTED ###
    checkpoint = torch.load(config["model"]["path"],
                            map_location="cpu",
                            weights_only=True)
    model = utils.instantiate_class(config["model"]["module"],
                                    config["model"]["class"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # IMPORTANT!

    ### EXPORT TO ONNX ###
    onnx_file_name = os.path.basename(config["model"]["path"]).split(".")[0] + ".onnx"
    onnx_output_path = os.path.join(config["output_path"], onnx_file_name)
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
