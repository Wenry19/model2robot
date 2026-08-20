
import numpy as np

from cuda.bindings import runtime as cudart

class TensorRTInference:

    """Manages TensorRT inference for fixed batch sizes, including buffers and CUDA resources."""

    def __init__(self, config, engine):

        self.config = config

        self.engine = engine

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Fixed input shape
        self.input_shape = (config["evaluation"]["batch_size"],
                            3,
                            config["model"]["input_height"],
                            config["model"]["input_width"])
        self.input_nbytes = (np.prod(self.input_shape) * np.dtype(np.float32).itemsize)

        # Set input shape and get output shape
        self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        # Allocate reusable host buffers
        self.host_output = np.empty(self.output_shape, dtype=np.float32)

        # Allocate reusable device buffers
        err, self.d_input = cudart.cudaMalloc(self.input_nbytes); self._check(err)
        err, self.d_output = cudart.cudaMalloc(self.host_output.nbytes); self._check(err)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # Create reusable CUDA stream
        err, self.stream = cudart.cudaStreamCreate(); self._check(err)

    def run(self, input_array):

        # Sanity checks and padding the batch if necessary
        original_batch_size = self._batch_padding(input_array)

        # Ensures float32 and contiguous memory, copies only if necessary
        host_input = np.ascontiguousarray(input_array, dtype=np.float32)

        # Host -> Device
        err, = cudart.cudaMemcpyAsync(
            self.d_input, host_input.ctypes.data, host_input.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        ); self._check(err)

        # TensorRT inference
        self.context.execute_async_v3(self.stream)

        # Device -> Host
        err, = cudart.cudaMemcpyAsync(
            self.host_output.ctypes.data, self.d_output, self.host_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        ); self._check(err)

        # Wait until all operations in the stream have completed
        err, = cudart.cudaStreamSynchronize(self.stream); self._check(err)

        # Keep only real samples
        outputs = self.host_output[:original_batch_size]
    
        return outputs

    def _batch_padding(self, input_array):

        original_batch_size = input_array.shape[0]
        inference_batch_size = self.config["evaluation"]["batch_size"]

        if original_batch_size > inference_batch_size:
            raise ValueError(
                f"Input batch size ({original_batch_size}) is larger than "
                f"the inference batch size ({inference_batch_size})"
            )

        if original_batch_size < inference_batch_size:
            padding = np.zeros(
                (
                    inference_batch_size - original_batch_size,
                    *input_array.shape[1:]
                ),
                dtype=input_array.dtype
            )

            input_array = np.concatenate([input_array, padding], axis=0)

        return original_batch_size

    def _check(self, err):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")

    def close(self):
        cudart.cudaFree(self.d_input)
        cudart.cudaFree(self.d_output)
        cudart.cudaStreamDestroy(self.stream)
