import torch
import math

def roi_find_match(thresholds, pairwise_matrix):
    """
    Args:
        thresholds (list[int]): holds two thresholds [0.5].
            It means that this function will label anchor with IoU overlap above 0.5 as positive,
            under 0.5 as negative.
        pairwise_matrix (Tensor): holds pairwise IoU overlap. Shape in N * M. ( N = #anchor, M = #gt_boxes )
    Returns:
        matches (Tensor[int64]): Vector of length N. i'th element stands for each anchor box.
            matches[i] = (matched gt box index, in value [0, M) )
        match_labels (Tensor[int8]): Vector of length N. i'th element stands for each anchor box.
            match_labels[i] = (weather a prediction(anchor to gt_box matching) is a true positive(>0.5), 
                               false positive(<0.5).)
    """
    
    assert pairwise_matrix.numel() != 0, "No gt box exists"
    assert torch.all(pairwise_matrix >= 0), "Negative IoU?"
    assert torch.all(pairwise_matrix <= 1), "IoU > 1?"



    # N * M pairwise matrix => .max(dim=1) gets the best gt_box candidate for each anchor prediction
    matched_vals, matches = pairwise_matrix.max(dim=1)
    # new tensor to store labels, initialized to 0

    match_labels = matches.new_full(matches.size(), 0, dtype=torch.int8)

    # Intervals
    # [0, 0.5)   => label 0,  false positive
    # [0.5, 1]   => label 1,  true positive
    for (label, (low, high)) in zip((0, 1), ((0, thresholds[0]), (thresholds[0], 1.001))):
        in_interval = (matched_vals >= low) & (matched_vals < high)
        match_labels[in_interval] = label

    return matches, match_labels


def roi_subsample_labels(
        matched_idxs,
        matched_labels, 
        batch_size_per_image=256, 
        positive_fraction=0.5,
        negative_label=0,
        positive_label=1
    ):
    """
    Based on the matching between N proposals and M groundtruth,
    sample the proposals and set their classification labels.

    Args:
        matched_idxs (Tensor): a vector of length N, each is the best-matched
            gt index in [0, M) for each proposal.
        matched_labels (Tensor): a vector of length N, 1 if positive, 0 if negative.
            automatically considerable as gt classes.
            1 if human, 0 if background

    Returns:
        Tensor: a vector of indices of sampled proposals. Each is in [0, N).
        Tensor: a vector of the same length, the classification label for
            each sampled proposal. Each sample is labeled as 0 if background, 1 if human.
    """
    positive_idx = (matched_labels == positive_label).nonzero(as_tuple=True)[0]
    negative_idx = (matched_labels == negative_label).nonzero(as_tuple=True)[0]

    num_pos = int(batch_size_per_image * positive_fraction)
    num_pos = min(positive_idx.numel(), num_pos)

    num_neg = batch_size_per_image - num_pos

    assert num_neg < negative_idx.numel(), "not enough negative samples"

    perm1 = torch.randperm(positive_idx.numel(), device=positive_idx.device)[:num_pos]
    perm2 = torch.randperm(negative_idx.numel(), device=negative_idx.device)[:num_neg]

    pos_idx = positive_idx[perm1]
    neg_idx = negative_idx[perm2]
    
    sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)

    # print(num_pos, num_neg, matched_idxs.device)
    return sampled_idxs, matched_labels[sampled_idxs]
    

def add_gt_to_proposals(proposals, gt_boxes):
    return torch.cat((proposals, gt_boxes))


def roi_get_gt_deltas(boxes, gt_boxes):
    """
    compute deltas needed to transform box into gt_box
    Args:
        boxes (Tensor): source boxes N * 4
        gt_boxes (Tensor): target of the transform N * 4
    Return:
        deltas (Tensor): transformation deltas N * (dx, dy, dw, dh)
    """
    src_widths = boxes[:, 2] - boxes[:, 0]
    src_heights = boxes[:, 3] - boxes[:, 1]
    src_ctr_x = boxes[:, 0] + 0.5 * src_widths
    src_ctr_y = boxes[:, 1] + 0.5 * src_heights

    target_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    target_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    target_ctr_x = gt_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = gt_boxes[:, 1] + 0.5 * target_heights

    wx, wy, ww, wh = (10.0, 10.0, 5.0, 5.0)
    dx = wx * (target_ctr_x - src_ctr_x) / src_widths
    dy = wy * (target_ctr_y - src_ctr_y) / src_heights
    dw = ww * torch.log(target_widths / src_widths)
    dh = wh * torch.log(target_heights / src_heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"

    return deltas

def roi_apply_deltas(boxes, deltas):
    """
    Perform anchor transformation defined in deltas.

    Args:
        deltas (Tensor): transformation deltas N * R * 4
        boxes (Tensor): boxes to be transformed R * 4
            where R is number of anchor box predictions in single feature map
            N is number of images in minibatch(2)
            (Anchors are common for every images in minibatch, since they are padded into same size)
    Return:
        transformed_boxes (Tensor): transformed boxes N * R * 4
    """
    boxes = boxes.unsqueeze(0)
    deltas = deltas.float()  # ensure fp32 for decoding precision
    boxes = boxes.to(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = (1.0, 1.0, 1.0, 1.0)
    dx = deltas[:, :, 0] / wx
    dy = deltas[:, :, 1] / wy
    dw = deltas[:, :, 2] / ww
    dh = deltas[:, :, 3] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, :, :1] = (pred_ctr_x - 0.5 * pred_w).unsqueeze(2)
    pred_boxes[:, :, 1:2] = (pred_ctr_y - 0.5 * pred_h).unsqueeze(2)
    pred_boxes[:, :, 2:3] = (pred_ctr_x + 0.5 * pred_w).unsqueeze(2)
    pred_boxes[:, :, 3:] = (pred_ctr_y + 0.5 * pred_h).unsqueeze(2)

    return pred_boxes.squeeze(0)