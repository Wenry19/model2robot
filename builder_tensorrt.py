
import sys
import os
import tensorrt as trt

import utils

def build_engine(onnx_path: str, engine_path: str):
    """Build a strongly typed TensorRT engine from an ONNX file and save it."""

    # Creating Logger and Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Creating a Network Definition
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    # Importing a Model Using the ONNX Parser
    # (the network definition must be populated from the ONNX representation)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    # Building an Engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)

if __name__ == "__main__":

    ### GET READY ###
    config_path = sys.argv[1]
    config = utils.read_config_file(config_path)
    utils.create_output_dir(config["output_path"])

    ### FOR REPRODUCIBILITY ###
    utils.save_config(config)

    ### BUILD TENSORRT ENGINE USING PYTHON API ###

    onnx_path = config["model"]["path"]
    engine_path = os.path.join(config["output_path"],
                               os.path.basename(onnx_path).split(".")[0] + ".engine")
    build_engine(onnx_path, engine_path)
