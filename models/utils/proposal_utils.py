"""
Anchors and NMS implemented here
"""
import math
import torch
from torchvision.ops import boxes as box_ops


def create_anchor_bases(sizes, aspect_ratios):
    anchor_bases = []

    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchor_bases.append([x0, y0, x1, y1])

    return torch.tensor(anchor_bases)


def create_anchors(anchor_bases, feature_shapes, strides=[4, 8, 16, 32, 64]):
    """
    Create anchor boxes for each scales.

    Returns list of torch.tensors in shape (Hi * Wi * (#anchor_bases = (#sizes * #ratios))) * 4.
    Length of return list is equal to the number of feature maps.
    """

    anchors = []

    for grid_size, stride, anchor_base in zip(feature_shapes, strides, anchor_bases):
        (grid_y, grid_x) = grid_size[-2:]

        shifts_x = torch.arange(stride/2, (grid_x + 0.5)*stride, step=stride, dtype=torch.float32, device=anchor_bases[0].device)
        shifts_y = torch.arange(stride/2, (grid_y + 0.5)*stride, step=stride, dtype=torch.float32, device=anchor_bases[0].device)

        # print(grid_x, grid_y)
        # print(grid_x * stride, grid_y * stride)
        # print(shifts_x)
        # print(shifts_y)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        # print(shift_x)
        # print(shift_y)
        # print(shift_x.shape)
        # print(shift_y.shape)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # print(shifts.shape)
        # print(anchor_base.shape)
        # print(anchor_base)

        anchor_stacks = []
        # for x in range(anchor_base.shape[0]):
        #     anchor_stacks.append(shifts + anchor_base[x])

        spread_shifts = torch.zeros([shifts.shape[0]*3, shifts.shape[1]], device=anchor_base.device)
        spread_shifts[0::3, :] = shifts
        spread_shifts[1::3, :] = shifts
        spread_shifts[2::3, :] = shifts

        spread_anchor_bases = torch.cat([anchor_base] * shifts.shape[0])

        anchor = spread_anchor_bases + spread_shifts

        # print(anchor.shape)
        # print(anchor_stacks[0][:3])
        # print(anchor_stacks[1][:3])
        # print(anchor_stacks[2][:3])
        # print("last anchor", anchor[-3:])
        anchors.append(anchor)

    return anchors


def get_pairwise_iou(boxes1, boxes2):
    """
    Compute the iou.
    Input boxes in form of (x1, y1, x2, y2)
    Args:
        boxes1, boxes2: Tensor, shape in N*4, M*4

    Returns:
        IoU intersection tensor, shape in N*M
    """

    area1 = compute_area(boxes1) # [N]
    area2 = compute_area(boxes2) # [M]

    inter = get_pairwise_intersections(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return iou



def get_pairwise_intersections(boxes1, boxes2):
    """
    intersecting area between every pairs of boxes.
    Args:
        boxes1, boxes2: Tensor, shape in N*4, M*4
    
    Returns:
        Tensor, in shape of N * M
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    lt = torch.min(boxes1[:, 2:].unsqueeze(1).expand(N, M, 2), boxes2[:, 2:].unsqueeze(0).expand(N, M, 2))
    rb = torch.max(boxes1[:, :2].unsqueeze(1).expand(N, M, 2), boxes2[:, :2].unsqueeze(0).expand(N, M, 2))
    intersection = torch.clamp(lt - rb, min=0)
    intersection = intersection.prod(dim=2)
    return intersection


def compute_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def find_match(thresholds, pairwise_matrix):
    """
    Args:
        thresholds (list[int]): holds two thresholds [0.3, 0.7].
            It means that this function will label anchor with IoU overlap above 0.7 as positive,
            between 0.3 upto 0.7 as ignored, under 0.3 as negative.
            Additionally, we sign positive label to the anchors with the highest IoU overlap for each gt boxes.
        pairwise_matrix (Tensor): holds pairwise IoU overlap. Shape in N * M. ( N = #anchor, M = #gt_boxes )
    Returns:
        matches (Tensor[int64]): Vector of length N. i'th element stands for each anchor box.
            matches[i] = (matched gt box index, in value [0, M) )
        match_labels (Tensor[int8]): Vector of length N. i'th element stands for each anchor box.
            match_labels[i] = (weather a prediction(anchor to gt_box matching) is a true positive(>0.7), 
                               false positive(<0.3), or ignored(0.3 < IoU < 0.7).)
    """
    
    assert pairwise_matrix.numel() != 0, "No gt box exists"
    assert torch.all(pairwise_matrix >= 0), "Negative IoU?"
    assert torch.all(pairwise_matrix <= 1), "IoU > 1?"

    # N * M pairwise matrix => .max(dim=1) gets the best gt_box candidate for each anchor prediction
    matched_vals, matches = pairwise_matrix.max(dim=1)

    # new tensor to store labels, initialized to 0
    match_labels = matches.new_full(matches.size(), 0, dtype=torch.int8)

    # Intervals
    # [0, 0.3)   => label 0,  false positive
    # [0.3, 0.7) => label -1, ignored
    # [0.7, 1]   => label 1,  true positive
    for (label, (low, high)) in zip((0, -1, 1), ((0, thresholds[0]), (thresholds[0], thresholds[1]), (thresholds[1], 1.001))):
        in_interval = (matched_vals >= low) & (matched_vals < high)
        match_labels[in_interval] = label

        # debug: positive anchors
        # if label == 1:
        #     print("\nlabel: ", label)  
        #     print("num match: ", match_labels[in_interval].numel())
        #     print("positive match values: ", matched_vals[in_interval])
        #     print("positive match indexes: ", in_interval.nonzero(as_tuple=True))

    # Faster R-CNN paper, 3.1.2. (i): the anchor with the highest IoI overlap with each gt_box 
    # For each gt_box, find the anchor prediction with highest match.
    highest_match_val_foreach_gt, _ = pairwise_matrix.max(dim=0)

    # Find anchors with found match value. It may be low, or it may be multiple anchors.
    # _, pred_inds_with_highest_match = (pairwise_matrix == highest_match_val_foreach_gt[None, :]).nonzero(as_tuple=True)

    # issue: 다 해결하긴 했는데, set_low_quality_matches_ -> 얘가 너~무 낮은 박스까지 포함시켜서 문제인듯함. 이거에 대한 threshold도 설정해줄 필요가 있을거같음.
    # 일반적인 코스로는 0.7 이상만 true인데, 이거도 최소한 0.5이상은 되어야하지않나. 근데 또 이걸 끄면 아예 샘플링이 안 되는것도 문제고.
    highest_match_val_foreach_gt = torch.where(highest_match_val_foreach_gt <= 0.3, -1.0, highest_match_val_foreach_gt.double())

    highest_matches = pairwise_matrix == highest_match_val_foreach_gt
    pred_inds_with_highest_match, _ = torch.nonzero(highest_matches, as_tuple=True)

    # put true positive label
    match_labels[pred_inds_with_highest_match] = 1

    # debug: positive anchors, zero IoU anchors
    # print("\n highest match val foreach gt")
    # print(highest_match_val_foreach_gt)
    # print(pred_inds_with_highest_match)
    # print(pred_inds_with_highest_match.shape)

    return matches, match_labels


def subsample_labels(
        labels, 
        batch_size_per_image=256, 
        positive_fraction=0.5,
        negative_label=0,
        positive_label=1,
        ignore_label=-1
    ):
    """
    Randomly sample a subset of positive and negative example, defined by arg :positive_fraction:.
    Overwrite the label vector to ignore value(-1), for all elements that are neither positive nor negative.

    Returns:
        labels (Tensor): in same size of input labels, modified in place.
    """
    positive_idx = (labels == positive_label).nonzero(as_tuple=True)[0]
    negative_idx = (labels == negative_label).nonzero(as_tuple=True)[0]

    num_pos = int(batch_size_per_image * positive_fraction)
    num_pos = min(positive_idx.numel(), num_pos)

    num_neg = batch_size_per_image - num_pos

    assert num_neg < negative_idx.numel(), "not enough negative samples"

    perm1 = torch.randperm(positive_idx.numel(), device=positive_idx.device)[:num_pos]
    perm2 = torch.randperm(negative_idx.numel(), device=negative_idx.device)[:num_neg]

    pos_idx = positive_idx[perm1]
    neg_idx = negative_idx[perm2]
    
    labels.fill_(-1)
    labels.scatter_(0, pos_idx, 1)
    labels.scatter_(0, neg_idx, 0)

    return pos_idx, neg_idx, labels



def get_gt_deltas(boxes, gt_boxes):
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

    wx, wy, ww, wh = (1.0, 1.0, 1.0, 1.0)
    dx = wx * (target_ctr_x - src_ctr_x) / src_widths
    dy = wy * (target_ctr_y - src_ctr_y) / src_heights
    dw = ww * torch.log(target_widths / src_widths)
    dh = wh * torch.log(target_heights / src_heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"

    return deltas


def apply_deltas(boxes, deltas):
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

    return pred_boxes

def clip(boxes, image_size):
    """
    Args:
        boxes (Tensor): shape in ((#boxes), 4)
        image_size (tuple[int]): (H, W)
    Return:
        boxes (Tensor): in place transform, fit into image size.
    """
    (h, w) = image_size
    boxes[:, :1] = torch.clamp(boxes[:, :1], min=0, max=w)
    boxes[:, 1:2] = torch.clamp(boxes[:, 1:2], min=0, max=h)
    boxes[:, 2:3] = torch.clamp(boxes[:, 2:3], min=0, max=w)
    boxes[:, 3:] = torch.clamp(boxes[:, 3:], min=0, max=h)
    return boxes


def apply_nms(boxes, scores_per_img, lvl, nms_thresh):
    return box_ops.batched_nms(boxes.float(), scores_per_img, lvl, nms_thresh)


def find_top_rpn_proposals(args, pred_proposals, pred_logits, image_sizes, is_training):
    """
    For each feature map, select the 'pre_nms_topk' highest scoring proposals,
    apply NMS, and clip proposals. Return the 'post_nms_topk'
    highest scoring proposals among all the feature maps for each image.
    Args:
        proposals (list[Tensor]): list length is number of feature maps.
            each list elements are transformed boxes in shape of (N, (Hi * Wi * 3), 4)
        pred_logits (list[Tensor]): list length is number of feature maps.
            each list elements are logits in shape of (N, (Hi * Wi * 3)).
        image_sizes (Tensor): (N, 2) tensor holding height and width of original image.

    Return:
        proposals (list[torch.Tensors]): i'th element contains 'post_nms_topk' object proposals for image i,
            sorted by their objectness score in descending order.
            list length is number of images in minibatch.
    """

    image_sizes = image_sizes.tolist()
    num_images = len(image_sizes)

    device = pred_proposals[0].device

    # Hyperparameters
    nms_thresh = args.rpn_nms_thresh
    pre_nms_topk = args.rpn_nms_topk_train if is_training else args.rpn_nms_topk_test
    post_nms_topk = args.rpn_nms_topk_post

    # Select 'pre_nms_topk' highest scoring proposals from each feature map (level)
    topk_scores = []
    topk_proposals = []

    # 셋이 나란히 append할건데, 나중에 cat했을때 어느 피쳐맵 레벨에서 나왔는지 확인하기 위함.
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)

    for level_id, (proposals_i, logits_i) in enumerate(zip(pred_proposals, pred_logits)):
        R = logits_i.shape[1]
        num_proposals_i = min(R, pre_nms_topk)

        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
        topk_idx = idx.narrow(1, 0, num_proposals_i)

        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i, ), level_id, dtype=torch.int64, device=device))
    
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    # Enumerate over images, apply NMS, choose topk results
    results = []

    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if is_training:
                raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]

        # clip boxes into image size
        boxes = clip(boxes, image_size)

        # filter empty boxes
        keep = torch.logical_not(torch.logical_or(boxes[:, 0] == boxes[:, 2], boxes[:, 1] == boxes[:, 3]))
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        scores_per_img, idx = scores_per_img.sort(descending=True)
        boxes = boxes[idx]
        keep = apply_nms(boxes, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]
        results.append(boxes[keep])
        # If objectness logits needed, then modify code to use scores_per_img[keep]

    return results
    

def predict_proposals(args, anchors, pred_logits, pred_reg_deltas, image_sizes, is_training):
    """
    Decode all the predicted box regression deltas to proposals. Find the top proposals
    by applying NMS and removing boxes that are too small.

    Args:
        anchors (list[torch.Tensors]): each element shape (Hi * Wi * (#anchor_bases(3) = (#sizes * #ratios))) * 4
            list length is equal to the number of feature maps. (default: 5)
        pred_reg_deltas (list[Tensor]): each element shape (N, Hi*Wi*(#anchor_bases(3)), (#box parametrization(4))),
            representing regression deltas used to transform anchors to proposals.
    Returns:
        proposals (list[torch.Tensors]): i'th element contains 'post_nms_topk' object proposals for image i,
            sorted by their objectness score in descending order.
            list length is number of images in minibatch.
    """
    with torch.no_grad():
        pred_proposals = [apply_deltas(anchor, pred_reg_delta) for (anchor, pred_reg_delta) in zip(anchors, pred_reg_deltas)]
        # Todo: find top 1000, NMS
        
        return find_top_rpn_proposals(args, pred_proposals, pred_logits, image_sizes, is_training)


def find_top_match_proposals(gt_boxes, proposals, image_id):
    """
    Args:
        proposals (Tensor): (1000, 4) tensor which holds proposal
        gt_boxes (Tensor): (N, 4) tensor which holds gt boxes
    Return:
        match (list[Tensor]): length N list, each element tensor holds top matches for each gt boxes.
    """
    topn = 3
    gt_boxes = torch.stack([x + torch.tensor([0, 0, x[0], x[1]], device=x.device) for x in gt_boxes], dim=0)
    pairwise_matrix = get_pairwise_iou(gt_boxes, proposals)
    sorted, indices = torch.sort(pairwise_matrix, descending=True, dim=1)

    # print("")
    # print(pairwise_matrix.shape)
    # print(image_id)
    # print(pairwise_matrix)
    # print(sorted)
    # print("")

    ret = []
    ret_prop = []
    for i in range(len(gt_boxes)):
        for j in range(topn):
            ret.append(proposals[indices[i][j]])
            ret_prop.append(pairwise_matrix[i][indices[i][j]])

    # print("")
    # print(image_id)
    # print(gt_boxes)
    # print(torch.stack(ret))
    # print(torch.stack(ret_prop))
    # print("")

    return torch.stack(ret), torch.stack(ret_prop)


def remove_zero_gt(gt_boxes):
    mask = torch.sum(gt_boxes, 1) != 0
    return gt_boxes[mask]


def test_anchor():
    sizes = [32, 64]
    aspect_ratios = [0.5, 1, 2]
    feature_shapes = [(24, 100, 100), (24, 50, 50)]
    anchor_bases = create_anchor_bases(sizes, aspect_ratios)
    create_anchors(anchor_bases, feature_shapes)


def test_iou():
    a = [
        [5,2,3],
        [7,9,6],
        [4,8,1]
    ]
    a = torch.tensor(a)

    # print(a)

    sorted, indices = a.sort(descending=True, dim=1)

    ret = []
    for i in range(3):
        for j in range(1):
            ret.append(indices[i][j])

    print(ret)


if __name__ == "__main__":
    test_iou()
