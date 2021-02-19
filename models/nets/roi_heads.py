import os
import torch
import time
from torch import nn
from torch.nn import functional as F
from models.nets.poolers import RoIPooler
from models.utils import proposal_utils as prop_ut
from models.utils import roi_head_utils as roi_ut

class ROIHeads(nn.Module):
    def __init__(
        self, 
        args,
        in_features=['p2', 'p3', 'p4', 'p5'],
        in_channels=256,
        pooler_resolution=7,
        pooler_scales=(1/4, 1/8, 1/16, 1/32),
        fc_dim=1024
    ):
        super().__init__()

        self.args = args
        self.in_features = in_features
        self.in_channels = in_channels
        self.pooler_resolution = pooler_resolution
        self.pooler_scales = pooler_scales

        # build pooler
        self.pooler = RoIPooler(output_size=pooler_resolution, scales=pooler_scales, method="RoIPool")  # RoIPool or RoIAlign

        # first fc 1024
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear((in_channels * pooler_resolution * pooler_resolution), fc_dim)
        self.fc_relu1 = nn.ReLU()

        # second fc 1024
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc_relu2 = nn.ReLU()

        # build predictor
        # one for human class, one for background
        self.cls_score = nn.Linear(fc_dim, 2)
        self.bbox_pred = nn.Linear(fc_dim, 4)

        # initialize
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)

        for l in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, gt_boxes):
        """
        Prepare training data in RoI head module.
        
        1. Assigns labels to proposals: IoU over 0.5 labeled positive
        2. Sample 512 boxes from proposals and gt boxes, with maximum positive fraction 0.25.
        *** Need to check if there is enough positive examples. If not, consider smaller batch size.
        *** Add print to check positive sample number.

        Args:
            same as forward.
        
        Return:
            sampled_proposals (list[Tensor]): 
                minibatch size length(2), each shape M'i x 4, M'i = samples per batch
            matched_gt_boxes (list[Tensor]): 
                minibatch size length(2), each shape M'i x 4, contains matched gt box for each positively labeled samples.
            labels (list[Tensor]): 
                minibatch size length(2), each elements are 1-d vectors, contains label 1 if positive, else 0.
        """
        # add gt to proposals
        proposals = [roi_ut.add_gt_to_proposals(proposal, gt_box) for (proposal, gt_box) in zip(proposals, gt_boxes)]

        sampled_proposals = []
        matched_gt_boxes = []
        labels = []

        for proposals_per_image, gt_boxes_per_image in zip(proposals, gt_boxes):
            pairwise_matrix = prop_ut.get_pairwise_iou(proposals_per_image, gt_boxes_per_image)

            # matcher: which box(0 ~ M-1), and matched labels(0, 1)
            matched_gt_box_idxs, matched_labels = roi_ut.roi_find_match([0.5], pairwise_matrix)
            del pairwise_matrix

            # sample
            sampled_idxs, gt_classes_per_image = roi_ut.roi_subsample_labels(
                matched_gt_box_idxs, 
                matched_labels, 
                batch_size_per_image=self.args.roi_batch_size, 
                positive_fraction=0.25
            )
            # append sampled training data

            sampled_proposal = proposals_per_image[sampled_idxs]
            matched_gt_box = gt_boxes_per_image[matched_gt_box_idxs[sampled_idxs]]
            label = matched_labels[sampled_idxs]

            sampled_proposals.append(sampled_proposal)
            matched_gt_boxes.append(matched_gt_box)
            labels.append(label)
        
        return sampled_proposals, matched_gt_boxes, labels


    def losses(self, cls_score, bbox_pred, proposal_boxes, gt_labels, gt_boxes):
        """
        calculate smooth l1 loss for bbox regression,
        cross entropy loss for classification.
        Args:
            cls_score (Tensor): M x 2 shape tensor, (logit_for_background, logit_for_human)
            bbox_pred (Tensor): M x 4 shape tensor, (dx, dy, dw, dh)
            proposal_boxes (Tensor): M x 4 shape tensor, (x1, y1, x2, y2)
            gt_labels (Tensor): M shape tensor, 0 for background, 1 for human
            gt_boxes (Tensor): M x 4 shape tensor, closest matching gt box of each proposal boxes.
                note that it is only for positive samples, equivalent to human class.
        Return:
            losses (Dict[str -> Loss]): holds "loss_cls", "loss_box_reg" losses
        """
        proposal_boxes = torch.cat(proposal_boxes)
        gt_labels = torch.cat(gt_labels).to(dtype=torch.int64)
        gt_boxes = torch.cat(gt_boxes)

        # calculate b-box regression loss
        # only for human class boxes
        pos_mask = gt_labels == 1
        neg_mask = gt_labels == 0

        gt_proposal_deltas = roi_ut.roi_get_gt_deltas(proposal_boxes[pos_mask], gt_boxes[pos_mask])

        loss_box_reg = F.smooth_l1_loss(
            bbox_pred[pos_mask],
            gt_proposal_deltas,
            0,
            reduction='sum'
        )

        # normalize by total number of proposal boxes
        loss_box_reg = loss_box_reg / gt_labels.numel()

        # calculate classification loss
        loss_cls = F.cross_entropy(cls_score, gt_labels, reduction="mean", weight=torch.Tensor((1, self.args.roi_pos_weight)).to(device=proposal_boxes.device))

        # extra stats
        softmax_scores = F.softmax(cls_score, dim=1)

        avg_pos_scores = softmax_scores[pos_mask][:, 1:]
        avg_neg_scores = softmax_scores[neg_mask][:, :1]

        avg_pos_scores = avg_pos_scores.mean()
        avg_neg_scores = avg_neg_scores.mean()

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}, avg_pos_scores, avg_neg_scores

    
    def inference(self, image_sizes, cls_scores, bbox_preds, proposal_boxes):
        """
        inferrence on human objects
        Args:
            image_sizes (list[Tensor]): list of (Hi, Wi)
            cls_score (Tensor): shape M x 2, 
                first half of dim 0 (M/2) for first image, second half for second image.
            bbox_pred (Tensor): shape M x 4,
                first half of dim 0 (M/2) for first image, second half for second image.
            proposal_boxes (list[Tensor]): list of length 2,
                tensor shape Mi x 4
        """

        # hyperparameters
        test_score_thresh = self.args.roi_test_score_thresh
        test_nms_thresh = self.args.roi_nms_thresh
        test_topk_per_image = self.args.roi_nms_topk_post
        
        # split into each images
        num_prop_per_image = [p.shape[0] for p in proposal_boxes]

        cls_scores = cls_scores.split(num_prop_per_image)
        bbox_preds = bbox_preds.split(num_prop_per_image)
        
        ret_boxes = []
        ret_scores = []
        ret_inds = []
        num_inferences = []

        for image_size, cls_score, bbox_pred, proposal_box in zip(image_sizes, cls_scores, bbox_preds, proposal_boxes):
            # predict_boxes using bbox_pred
            boxes = roi_ut.roi_apply_deltas(proposal_box, bbox_pred)

            # predict_probs
            scores = F.softmax(cls_score, dim=1)

            valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]

            scores = scores[:, -1:]
            num_bbox_reg_classes = 1

            #clip boxes into image shape
            boxes = prop_ut.clip(boxes, image_size)
            boxes = boxes.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
            filter_mask = scores > test_score_thresh  # R x K
            filter_inds = filter_mask.nonzero()
            boxes = boxes[filter_inds[:, 0], 0]
            scores = scores[filter_mask]

            keep = prop_ut.apply_nms(boxes, scores, filter_inds[:, 1], test_nms_thresh)
            keep = keep[:test_topk_per_image]
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

            ret_boxes.append(boxes)
            ret_scores.append(scores)
            ret_inds.append(filter_inds[:, 0])

            num_inference = keep.shape[0]
            num_inferences.append(num_inference)
        
        ret_boxes = torch.cat(ret_boxes)
        ret_scores = torch.cat(ret_scores)
        ret_inds = torch.cat(ret_inds)
        num_inferences = torch.Tensor(num_inferences).to(bbox_pred.device, dtype=torch.int)
        
        return (ret_boxes, ret_scores, ret_inds, num_inferences)


    def forward(self, batched_imgs, image_sizes, feature_maps, proposals, annotations, is_training=True):
        """
        Args:
            batched_imgs (Tensor): shape N x C x H x W
            feature_maps (list[Tensor]): length 5, shape N x C x Hi x Wi
            proposals (list[Tensor]): minibatch size length(2), each shape Mi x 4
            gt_boxes (list[Tensor]): minibatch size length(2), each shape Gi x 4

        Forward in order:
            label and sample proposals -> 여기서도 training example 샘플링함.

            forward_box
                box feature pooling
                box feature through head: 1024 fc, 1024 fc
                box feature into prediction network (linear layers)
                    output: scores and deltas

            if train, get loss
            if eval, predict instances
        """
        gt_boxes = [[torch.tensor([x[0], x[1], x[2], x[3]], device=x.device) for x in annotation if not (x[0] == 0 and x[1] == 0 and x[2] == 0 and x[3] == 0)] for annotation in annotations]
        gt_boxes = [torch.stack([x + torch.tensor([0, 0, x[0], x[1]], device=x.device) for x in gt_box], dim=0) for gt_box in gt_boxes]

        # label and sample proposals

        if is_training:
            sampled_proposals, matched_gt_boxes, sampled_labels = self.label_and_sample_proposals(proposals, gt_boxes)
        else:
            sampled_proposals = proposals
        
        feature_maps = [feature_maps[f] for f in self.in_features]
        box_features = self.pooler(feature_maps, sampled_proposals)

        box_features = self.flat(box_features)
        box_features = self.fc1(box_features)
        box_features = self.fc_relu1(box_features)
        box_features = self.fc2(box_features)
        box_features = self.fc_relu2(box_features)

        cls_score = self.cls_score(box_features)
        bbox_pred = self.bbox_pred(box_features)

        if is_training:
            losses, pos_score, neg_score = self.losses(cls_score, bbox_pred, sampled_proposals, sampled_labels, matched_gt_boxes)
            inference = ()
            extra = (pos_score, neg_score)
        else:
            inference = self.inference(image_sizes, cls_score, bbox_pred, sampled_proposals)
            losses = {}
            extra = ()
        
        return losses, inference, extra
