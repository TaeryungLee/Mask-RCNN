import os
import pickle
import requests
from pprint import pprint
import torch
from torch import nn
import numpy as np
from models.nets.resnet import ResNet
from models.nets.fpn import FPN
from models.utils.pretrained_weight_matcher import matcher

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

    if args.is_train:
        if args.load_pretrained_resnet:
            bottom_up, match_dict = load_pretrained_resnet(bottom_up, args.pretrained_resnet)

    in_features = bottom_up.out_features
    out_channels = args.fpn_out_chan

    # top_block = LastLevelMaxPool()
    # fuse_type = "sum"

    if args.model_name == "ResNet-50-FPN":
        return FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, top_block=True), match_dict


def load_pretrained_resnet(model, filename):
    link = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl"

    if not os.path.isfile(filename):
        print("downloading pretrained model")
        r = requests.get(link)
        open(filename, 'wb').write(r.content)
        print("successfully downloaded pretrained model in ", filename)

    
    with open(filename, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    pretrained_state_dict = data
    model_state_dict = model.state_dict()
    mats = {}

    for key in pretrained_state_dict.keys():
        mat = matcher(key)
        if mat is not None:
            if mat not in model_state_dict.keys():
                print(mat, "not in my model....")
                raise ValueError("something wired happend")
            
            if mat in mats.keys():
                print(mat, "is already in mats")
                raise ValueError("something wired happened")
            
            mats[key] = mat

        else:
            pass
    

    dict_to_feed = {}

    for key in mats.keys():
        dict_to_feed[mats[key]] = torch.tensor(pretrained_state_dict[key])
    
    model_state_dict.update(dict_to_feed)

    model.load_state_dict(model_state_dict)
    
    return model, mats