
import sys
import os

from model2robot.inference.utils import build_tensorrt_engine
import model2robot.utils as utils

def main():

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
    build_tensorrt_engine(onnx_path, engine_path)

if __name__ == "__main__":
    main()
