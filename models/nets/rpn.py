import torch
from torch import nn
from torch.nn import functional as F
from ..utils import proposal_utils as ut


class RPN(nn.Module):
    def __init__(
        self,
        in_features=['p2', 'p3', 'p4', 'p5', 'p6'],
        batch_size_per_image=256,
        positive_fraction=0.5,
        negative_under_fraction=0.1,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
        anchor_boundary_thresh=-1.0,
        loss_weight=1.0,
        box_reg_loss_type="smooth_l1",
        smooth_l1_beta=0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = RPNHeads()

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.negative_under_fraction = negative_under_fraction

        self.pre_nms_topk = {"train": pre_nms_topk[0], "test": pre_nms_topk[1]}
        self.post_nms_topk = {"train": post_nms_topk[0], "test": post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = 0
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

        self.anchor_bases = [ut.create_anchor_bases([size], [0.5, 1, 2]) for size in [32, 64, 128, 256, 512]]
    
    def loss(
        self,
        anchors,
        pred_logits,
        gt_labels,
        pred_reg_deltas,
        gt_boxes
    ):
        """
        Calculate loss from predicted logits and regression deltas.
        Each input argument should be in format:
            anchors (list[])
        """
        pass

    def forward(self, features, image_sizes, annotations):
        # build anchors
        features = [features[key] for key in self.in_features]
        feature_shapes = [x.shape[-2:] for x in features]
        

        anchors = ut.create_anchors(self.anchor_bases, feature_shapes)

        pred_logits, pred_reg_deltas = self.rpn_head(features)
        # print(" ")
        # print("device", features[0].device)
        # print("feature shapes: ", feature_shapes)
        # print("logits 2", pred_logits[0].shape)
        # print("logits 3", pred_logits[1].shape)
        # print("logits 4", pred_logits[2].shape)
        # print("logits 5", pred_logits[3].shape)
        # print("logits 6", pred_logits[4].shape)

        # print("deltas 2", pred_reg_deltas[0].shape)
        # print("deltas 3", pred_reg_deltas[1].shape)
        # print("deltas 4", pred_reg_deltas[2].shape)
        # print("deltas 5", pred_reg_deltas[3].shape)
        # print("deltas 6", pred_reg_deltas[4].shape)
        



        return pred_logits, pred_reg_deltas

class RPNHeads(nn.Module):
    def __init__(
        self,
        in_channels=256,
        num_anchors=3,
        reg_delta_dim=4
    ):
        super().__init__()

        self.inter_layer = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.reg_deltas = nn.Conv2d(in_channels, num_anchors*reg_delta_dim, kernel_size=1, stride=1)

        for l in [self.inter_layer, self.logits, self.reg_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
    
    def forward(self, features):

        pred_logits = []
        pred_reg_deltas = []

        for x in features:
            inter = self.inter_layer(x)
            inter = F.relu(inter)
            logit = self.logits(inter)
            reg_delta = self.reg_deltas(inter)

            pred_logits.append(logit)
            pred_reg_deltas.append(reg_delta)
        
        return pred_logits, pred_reg_deltas
            


def test():
    rpn = RPN()



