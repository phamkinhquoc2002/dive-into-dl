import torch
import numpy as numpy
from torch import nn
from d2l import torch as d2l
from config import init_cnn

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(*conv_blks, nn.Flatten(), 
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5), 
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5), 
                                 nn.LazyLinear(num_classes)
                                 )
        self.net.apply(init_cnn)

if __name__ == '__main__':
    model = VGG(lr = 0.1, arch=(3, 32))
    data = d2l.FashionMNIST(batch_size=128, resize=(32, 32))
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.fit(model, data)
