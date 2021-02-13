import torch
import os
from torch import nn
import requests
import pickle
from .nets.rpn import RPN
from models.utils.pretrained_weight_matcher import special_cases, matcher

def build_proposal_generator(args):
    return RPN(args)
