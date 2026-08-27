
import torch

from inference.inference import Inference
from utils import instantiate_class

class PyTorchInference(Inference):

    def __init__(self, checkpoint_path, model_module, model_class, device):

        self.device = device

        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                weights_only=True)

        self.model = instantiate_class(model_module,
                                       model_class).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def run(self, input_array):

        input_array = input_array.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_array)

        return outputs

    def close(self):
        pass
