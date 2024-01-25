import torch
from torchvision import datasets, transforms


# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


