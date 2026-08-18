
from PIL import Image
from torch.utils.data import Dataset


class DoorDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Load label
        label = self.labels[idx]

        return image, label
