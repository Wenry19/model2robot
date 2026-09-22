
import sys
import os
from datetime import datetime, timezone

from model2robot.inference.utils import build_tensorrt_engine
import model2robot.utils as utils

def main():

    started_at = datetime.now(timezone.utc)

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

    utils.make_directory_only_read(config["output_path"])

    finished_at = datetime.now(timezone.utc)

    ### EXPERIMENT MANIFEST ###
    
    utils.generate_experiment_manifest(started_at=started_at,
                                       finished_at=finished_at,
                                       input_artifact_path=os.path.dirname(config["model"]["path"]),
                                       output_artifact_path=config["output_path"],
                                       experiment_type="build_tensorrt",
                                       output_manifest_path=config["manifest_path"])

if __name__ == "__main__":
    main()
