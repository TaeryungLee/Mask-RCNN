import torch
from models.nets.roi_heads import ROIHeads

def build_roi_heads(args):
    return ROIHeads(args)