from models.nets.resnet import ResNet
from torch import nn
from torch.nn import functional as F
import torch

class FPN(nn.Module):
    def __init__(self, bottom_up, in_features, out_channels, top_block=True):
        """
        Args:
            bottom_up: bottom up network module.(ResNet)
            in_features: input feature layer names.
            out_channels: output channels of this network

        """
        super().__init__()
        # bottom_up outputs
        self.bottom_up = bottom_up
        self.in_features = in_features
        self.in_channels = bottom_up._out_channels
        self.in_strides = bottom_up.strides

        # output channel
        self.out_channel = out_channels

        # initialize layers
        # self.C1 = bottom_up.C1
        # self.C2 = bottom_up.C2
        # self.C3 = bottom_up.C3
        # self.C4 = bottom_up.C4
        # self.C5 = bottom_up.C5

        if top_block:
            self.P6_pool = nn.MaxPool2d(kernel_size=1, stride=2)
        else:
            self.P6_pool = None

        # self.P(i)_conv(j): j = 1 or 2,
        # if j = 1: lateral conv
        # if j = 2: output conv
        
        self.P5_conv1 = nn.Conv2d(self.in_channels[-1], self.out_channel, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

        self.P4_conv1 = nn.Conv2d(self.in_channels[-2], self.out_channel, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

        self.P3_conv1 = nn.Conv2d(self.in_channels[-3], self.out_channel, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

        self.P2_conv1 = nn.Conv2d(self.in_channels[-4], self.out_channel, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

    def output_shape(self):
        """
        channels and strides for each out feature names Dict[str to Dict[str to int]]
        ex) {"P2": {"channels": 256, "strides": 4}, 
             "P3": {"channels": 256, "strides": 8}, ... }
        """
        return {"P{}".format(i+2): 
            {"channels": self.out_channel, "strides": self.in_strides[i]} for i in range(4)}
        

    def forward(self, x):
        with torch.no_grad():
            bottom_up_features = self.bottom_up(x)
        
        # bottom_up_outputs = [bottom_up_features["res{}".format(i)] for i in range(5, 1, -1)]

        # P5
        C5 = bottom_up_features['res5']
        P5 = self.P5_conv1(C5)
        P5 = self.P5_conv2(P5)

        # P6
        if self.P6_pool is not None:
            P6 = self.P6_pool(P5)
        
        # P4
        C4 = bottom_up_features['res4']
        P4 = self.P4_conv1(C4) + F.interpolate(P5, scale_factor=2, mode="nearest")
        P4 = self.P4_conv2(P4)

        # P3
        C3 = bottom_up_features['res3']
        P3 = self.P3_conv1(C3) + F.interpolate(P4, scale_factor=2, mode="nearest")
        P3 = self.P3_conv2(P3)

        # P2
        C2 = bottom_up_features['res2']
        P2 = self.P2_conv1(C2) + F.interpolate(P3, scale_factor=2, mode="nearest")
        P2 = self.P2_conv2(P2)

        if self.P6_pool is not None:
            return {"p2": P2, "p3": P3, "p4": P4, "p5": P5, "p6": P6}
        else:
            return {"p2": P2, "p3": P3, "p4": P4, "p5": P5}
    