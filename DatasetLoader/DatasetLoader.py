import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    """
    Custom MNIST dataset using torchvision.
    This class automatically downloads the dataset and applies transformations.
    """

    def __init__(self, train=True):
        """
        :param train: Boolean flag to specify training (True) or test (False) dataset.
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        self.dataset = torchvision.datasets.MNIST(
            root='./data',  # Download location
            train=train, 
            transform=self.transform,
            download=True  # Automatically downloads if not available
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
        - Image tensor of shape (1, 28, 28)
        - Corresponding label (integer between 0-9)
        """
        im_tensor, label = self.dataset[index]
        return im_tensor


class CIFARDataset(Dataset):
    """
    Custom Cifar dataset using torchvision.
    This class automatically downloads the dataset and applies transformations.
    """

    def __init__(self, train=True):
        """
        :param train: Boolean flag to specify training (True) or test (False) dataset.
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',  # Download location
            train=train, 
            transform=self.transform,
            download=True  # Automatically downloads if not available
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
        - Image tensor of shape (1, 28, 28)
        - Corresponding label (integer between 0-9)
        """
        im_tensor, label = self.dataset[index]
        return im_tensor