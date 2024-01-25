import torch
from torch import nn

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Weight initialization
def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

