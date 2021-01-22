import torch
from torch import nn
from models.nets.resnet import ResNet
from models.nets.fpn import FPN

def build_backbone(args):
    """
    Builds backbone network as defined in args.
    Default is ResNet-FPN backbone.

    Returns nn.Module class.
    """

    if args.model_name == "ResNet-50-FPN":
        # todo: get depth, out_features, bn from args and put to initialization of ResNet.
        bottom_up = ResNet()
    
    else:
        raise NotImplementedError("no such model")

    in_features = bottom_up.out_features
    out_channels = args.fpn_out_chan

    # top_block = LastLevelMaxPool()
    # fuse_type = "sum"

    if args.model_name == "ResNet-50-FPN":
        return FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, top_block=True)
