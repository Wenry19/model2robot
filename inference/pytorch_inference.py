
import torch

from inference.inference import Inference
from models.detectdoor import Detectdoor

class PyTorchInference(Inference):

    def __init__(self, checkpoint_path, device):

        self.device = device

        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                weights_only=True)

        self.model = Detectdoor().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def run(self, input_array):

        input_array = input_array.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_array)

        return outputs

    def close(self):
        pass
