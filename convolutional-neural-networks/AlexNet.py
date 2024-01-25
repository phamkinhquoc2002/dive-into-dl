import torch
import numpy as numpy
from torch import nn
from d2l import torch as d2l
from config import device, init_cnn


class AlexNet(d2l.Classifier):
    def __init__(self, lr = 0.1, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=4, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride = 2), nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LazyLinear(4096),
            nn.LazyLinear(num_classes)
        )
        self.net.apply(init_cnn)
        self.to(device)

if __name__ == '__main__':
    model = AlexNet(lr = 0.1)
    data = d2l.FashionMNIST(batch_size=128, resize=(32, 32))
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.fit(model, data)