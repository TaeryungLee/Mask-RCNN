import torch
import os
from torch import nn
import requests
import pickle
from .nets.rpn import RPN
from models.utils.pretrained_weight_matcher import special_cases, matcher

def build_proposal_generator(args):
    model = RPN()
    model, _ = load_rpn(model)

    return model



def load_rpn(model):
    link = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl"
    filename = "pretrained/pick.pkl"
    if not os.path.isfile(filename):
        r = requests.get(link)
        open(filename, 'wb').write(r.content)

    with open(filename, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    pretrained_state_dict = data["model"]

    model_state_dict = model.state_dict()

    mats = {}

    for key in pretrained_state_dict.keys():
        mat = matcher(key)
        if mat is not None and mat in special_cases.keys():
            mats[key] = mat
        
        else:
            pass

    dict_to_feed = {}

    for key in mats.keys():
        dict_to_feed[mats[key]] = torch.tensor(pretrained_state_dict[key])
    
    model_state_dict.update(dict_to_feed)

    model.load_state_dict(model_state_dict)
    return model, mats