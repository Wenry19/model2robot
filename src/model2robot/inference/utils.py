
import tensorrt as trt

def build_tensorrt_engine(onnx_path: str, engine_path: str):
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
