
import torch

from model2robot.inference.inference import Inference

class PyTorchInference(Inference):

    def __init__(self, model_path, model_instance, device):

        self.model = model_instance
        self.device = device

        model_info = torch.load(model_path,
                                map_location=self.device,
                                weights_only=True)

        self.model.load_state_dict(model_info["model_state_dict"])
        self.model.eval()

    def run(self, input_array):

        input_array = input_array.to(self.device)

        with torch.no_grad():
            self.outputs = self.model(input_array)

    def synchronize(self):
        # Wait for all pending CUDA operations to finish.
        # Needed for accurate inference timing during benchmarking.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_output(self):
        self.synchronize()
        return self.outputs.cpu()

    def close(self):
        pass
