
from torchvision import transforms

def get_default_transform(input_height, input_width):

    return transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor()
    ])