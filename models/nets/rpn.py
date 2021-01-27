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
        
    
    def losses(
        self,
        anchors,
        pred_logits,
        gt_labels,
        pred_reg_deltas,
        gt_boxes,
        image_sizes
    ):
        """
        Calculate loss from predicted logits and regression deltas.
        Each input argument should be in format:
            anchors (list[torch.Tensors]): each element shape (Hi * Wi * (#anchor_bases(3) = (#sizes * #ratios))) * 4
                list length is equal to the number of feature maps. (default: 5)
            pred_logits (list[Tensor]): each element shape (N, Hi*Wi*(#anchor_bases)), 
                representing the predicted objectness logits, for all anchors.
                list length is equal to the number of feature maps.
            pred_reg_deltas (list[Tensor]): each element shape (N, Hi*Wi*(#anchor_bases(3)), (#box parametrization(4))),
                representing regression deltas used to transform anchors to proposals.
            gt_labels (list[Tensor]): output gt_labels of label_and_sample_anchors
            gt_boxes (list[Tensor]): output matched_gt_boxes of label_and_sample_anchors 
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        pos_mask = gt_labels == 1
        neg_mask = gt_labels == 0

        # num_pos_anchors = pos_mask.sum().item()
        # num_neg_anchors = neg_mask.sum().item()

        merged_anchors = torch.cat(anchors, dim=0)
        gt_anchor_deltas = [ut.get_gt_deltas(merged_anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)



        loc_loss = F.smooth_l1_loss(
            torch.cat(pred_reg_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self.smooth_l1_beta,
            reduction='sum'
        )

        valid_mask = gt_labels >= 0
        obj_loss = F.binary_cross_entropy_with_logits(
            torch.cat(pred_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction='sum'
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": obj_loss / normalizer,
            "loss_rpn_loc": loc_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors, annotations, image_sizes):
        """
        Args:
            anchors (list[torch.Tensors]): each element shape (Hi * Wi * (#anchor_bases = (#sizes * #ratios))) * 4
                list length is equal to the number of feature maps. (default: 5)
                sum((Hi * Wi * 1(size) * 3(ratio))
                -> commonly used for every input images. It is used only to identify anchor boxes.
            annotations (list[torch.Tensors]): each element contains bounding box for single image in minibatch.
                list length is equal to minibatch size.(#img)
            image_sizes (torch.Tensors): [(im.shape[-2], im.shape[-1]) for im in images] into tensor
        Returns:
            gt_labels (list[torch.Tensors]): 
                list length is equal to minibatch size(#img).
                each elements hold labels for anchors across every feature maps sum((Hi * Wi * 1(size) * 3(ratio)).
                So, each elements are 1-d vector of length sum((Hi * Wi * 1(size) * 3(ratio)).
                label values mean: -1 = ignore, 0 = negative, 1 = positive
            matched_gt_boxes (list[torch.Tensors]): 
                list length is equal to minibatch size(#img).
                each elements are R*4 tensors. R = sum((Hi * Wi * 1(size) * 3(ratio))
                i-th element holds matched ground truth box for R anchors.
                Values are only assigned for positive labeled anchors.
        """
        # list[list[Tensor]]

        gt_boxes = [[x for x in annotation if not (x[0] == 0 and x[1] == 0 and x[2] == 0 and x[3] == 0)] for annotation in annotations]
        gt_boxes = [torch.stack([x + torch.tensor([0, 0, x[0], x[1]], device=x.device) for x in gt_box], dim=0) for gt_box in gt_boxes]
        
        # original size info
        # anchor_size[0] = #feature maps
        # anchor_size[1][i] = #anchors for i'th feature maps, (Hi * Wi * 3(ratio) * 1(size))
        # gt_box_size[0] = #images in minibatch
        # gt_box_size[1][i] = for i'th image, # of ground truth boxes
        anchor_size = (len(anchors), (anchors[0].shape[0], anchors[1].shape[0], anchors[2].shape[0], anchors[3].shape[0], anchors[4].shape[0]))
        gt_box_size = (len(gt_boxes), (gt_boxes[0].shape[0], gt_boxes[1].shape[0]))


        # Concatenate both anchors and gt_boxes into 2-d tensor (N * 4)
        merged_anchor = torch.cat(anchors, dim=0)
        merged_gt_boxes = torch.cat(gt_boxes, dim=0)

        # print(anchors, gt_boxes)
        # print(anchor_size, gt_box_size)
        # print(merged_anchor.shape, merged_gt_boxes.shape)
        # print(merged_anchor, merged_gt_boxes)
        
        gt_labels = []
        matched_gt_boxes = []

        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            # Issue: 여기 들어가는 게 merged_gt_boxes면 안될거같다.
            # pairwise_iou = ut.get_pairwise_iou(merged_anchor, merged_gt_boxes)

            pairwise_iou = ut.get_pairwise_iou(merged_anchor, gt_boxes_i)
            # matcher
            # iou threshold = 0.3, 0.7
            # iou labels = 0, -1, 1
            matched_idxs, gt_labels_i = ut.find_match([0.3, 0.7], pairwise_iou)
            del pairwise_iou

            pos_idx, neg_idx, gt_labels_i = ut.subsample_labels(
                gt_labels_i, 
                batch_size_per_image=self.batch_size_per_image, 
                positive_fraction=self.positive_fraction,
                negative_label=0,
                positive_label=1,
                ignore_label=-1
            )

            # debug: positive anchors
            # print("\n positive boxes \n")

            # for pos_id in pos_idx:
            #     print(pos_id)
            #     print(merged_anchor[pos_id])
            #     print(pairwise_iou[pos_id:pos_id + 1,:])

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway.
                matched_gt_boxes_i = torch.zeros_like(merged_anchor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs]

            # print("shapes")
            # print(matched_gt_boxes_i.shape)
            # print(gt_labels_i.shape)
            # print(torch.nonzero(matched_gt_boxes_i).numel())

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def forward(self, features, image_sizes, annotations):
        """
        bbox annotations = (x1, y1, delta_x, delta_y) => x2 = x1 + delta_x, y same.
        anchor annotations = (x1, y1, x2, y2)
        """
        # build anchors
        features = [features[key] for key in self.in_features]
        feature_shapes = [x.shape[-2:] for x in features]
        
        self.anchor_bases = [x.to(features[0].device) for x in self.anchor_bases]
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


        # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        pred_logits = [
            score.permute(0, 2, 3, 1).flatten(1) for score in pred_logits
        ]

        # print(pred_reg_deltas[0].shape)
        # print(pred_reg_deltas[1].shape)

        # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        pred_reg_deltas = [
            x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2) for x in pred_reg_deltas
        ]

        # print(pred_reg_deltas[0].shape)
        # print(pred_reg_deltas[1].shape)

        if annotations is not None:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, annotations, image_sizes)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_reg_deltas, gt_boxes, image_sizes)
        else:
            losses = None
    
        proposals = ut.predict_proposals(anchors, pred_logits, pred_reg_deltas, image_sizes)

        return proposals, losses

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



