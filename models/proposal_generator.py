import torch
from torch import nn
from .nets.rpn import RPN

def build_proposal_generator(args):
    return RPN()