from abc import ABC, abstractmethod


class Inference(ABC):

    @abstractmethod
    def run(self, input_array):
        """Launch inference on a batch of inputs."""
        pass

    @abstractmethod
    def synchronize(self):
        """Wait for all pending inference operations to finish."""
        pass

    @abstractmethod
    def get_output(self):
        """Return inference output as a CPU PyTorch tensor."""
        pass

    @abstractmethod
    def close(self):
        """Release resources allocated for inference."""
        pass