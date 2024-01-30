import torch
import numpy as numpy
from torch import nn
from d2l import torch as d2l
from config import init_cnn

def nin_block(out_channels, kernel_size, stride, padding):
    return nn.Sequantial(
        nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU()
    )

class NiN(d2l.Module):
    def __init__(self, lr =0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(out_channels=96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(out_channels=256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(out_channels=384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            nin_block(out_channels=num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveMaxPool2d(1,1)
            nn.Flatten()
            )
        self.net.apply(init_cnn)

if __name__ == '__main__':
     model = NiN(lr=0.05)
     trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
     data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
     model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn())
     trainer.fit(model, data)



