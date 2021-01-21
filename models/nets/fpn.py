from models.nets.resnet import ResNet
from torch import nn
from torch.nn import functional as F
import torch

class FPN(nn.Module):
    def __init__(self, bottom_up, in_features, out_channels, top_block=None, fuse_type=None):
        """
        Args:
            bottom_up: bottom up network module.(ResNet)
            in_features: input feature layer names.
            out_channels: output channels of this network

        """
        super().__init__()

    def output_shape(self):
        """
        channels and strides for each out feature names Dict[str to Dict[str to int]]
        ex) {"P2": {"channels": 256, "strides": 4}, 
             "P3": {"channels": 256, "strides": 8}, ... }
        """
        pass

    def forward(self, x):
        pass

    