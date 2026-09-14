
import numpy as np
import torch

from inference.inference import Inference

from cuda.bindings import runtime as cudart
import tensorrt as trt

class TensorRTInference(Inference):

    """Manages TensorRT inference for fixed batch sizes, including buffers and CUDA resources."""

    def __init__(self, model_path, batch_size, input_height, input_width):

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(model_path, "rb") as f:
            model_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(model_data)

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Fixed input shape
        self.input_shape = (self.batch_size,
                            3,
                            self.input_height,
                            self.input_width)
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

        # Ensures float32 and contiguous memory, copies only if necessary
        host_input = np.ascontiguousarray(input_array, dtype=np.float32)

        # Sanity checks and padding the batch if necessary
        self.original_batch_size = self._batch_padding(host_input)

        # Host -> Device
        err, = cudart.cudaMemcpyAsync(
            self.d_input, host_input.ctypes.data, host_input.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        ); self._check(err)

        # TensorRT inference
        self.context.execute_async_v3(self.stream)

    def _batch_padding(self, input_array):

        original_batch_size = input_array.shape[0]
        inference_batch_size = self.batch_size

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

    def synchronize(self):
        # Wait for all pending operations in the CUDA stream to finish.
        # Needed for accurate inference timing during benchmarking.
        err, = cudart.cudaStreamSynchronize(self.stream)
        self._check(err)

    def get_output(self):

        # Device -> Host
        err, = cudart.cudaMemcpyAsync(
            self.host_output.ctypes.data, self.d_output, self.host_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        ); self._check(err)

        self.synchronize()

        # Keep only real samples
        outputs = self.host_output[:self.original_batch_size]

        # Convert TensorRT NumPy output to PyTorch tensor
        outputs = torch.from_numpy(outputs)
    
        return outputs

    def close(self):
        cudart.cudaFree(self.d_input)
        cudart.cudaFree(self.d_output)
        cudart.cudaStreamDestroy(self.stream)
