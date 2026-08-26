from abc import ABC, abstractmethod


class Inference(ABC):

    @abstractmethod
    def run(self, input_array):
        """Run inference on a batch of inputs."""
        pass

    @abstractmethod
    def close(self):
        pass