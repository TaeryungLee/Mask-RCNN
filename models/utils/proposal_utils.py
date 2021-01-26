"""
Anchors and NMS implemented here
"""
import math
import torch


def create_anchor_bases(sizes, aspect_ratios):
    anchor_bases = []

    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchor_bases.append([x0, y0, x1, y1])

    return torch.tensor(anchor_bases).cuda()


def create_anchors(anchor_bases, feature_shapes, strides=[2, 4, 8, 16, 32]):
    """
    Create anchor boxes for each scales.

    Returns list of torch.tensors in shape (Hi * Wi * (#anchor_bases = (#sizes * #ratios))) * 4.
    Length of return list is equal to the number of feature maps.
    """
    # Additional args
    strides = [2, 4, 8, 16, 32]

    anchors = []

    for grid_size, stride, anchor_base in zip(feature_shapes, strides, anchor_bases):
        (grid_x, grid_y) = grid_size[-2:]
        shifts_x = torch.arange(0, grid_x*stride, step=stride, dtype=torch.float32, device=anchor_bases[0].device)
        shifts_y = torch.arange(0, grid_y*stride, step=stride, dtype=torch.float32, device=anchor_bases[0].device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # print(shift_x.shape)
        # print(shift_y.shape)
        # print(shifts.shape)
        # print(anchor_bases.shape)

        anchor_stacks = []
        for x in range(anchor_base.shape[0]):
            anchor_stacks.append(shifts + anchor_base[x])
        
        anchor = torch.cat(anchor_stacks, dim=0)
        # print(anchor.shape)
        # print(anchor_stacks[0][:3])
        # print(anchor_stacks[1][:3])
        # print(anchor_stacks[2][:3])
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
    match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

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
    highest_match_val_foreach_gt = torch.where(highest_match_val_foreach_gt <= 0.5, -1.0, highest_match_val_foreach_gt.double())

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



def compute_refinement(box, gt_box):
    """
    compute refinement needed to transform box to gt_box
    """
    pass


def do_regression(anchors, deltas):
    """
    Perform anchor transformation defined in deltas.
    """
    pass



def test_anchor():
    sizes = [32, 64]
    aspect_ratios = [0.5, 1, 2]
    feature_shapes = [(24, 100, 100), (24, 50, 50)]
    anchor_bases = create_anchor_bases(sizes, aspect_ratios)
    create_anchors(anchor_bases, feature_shapes)



if __name__ == "__main__":
    test_anchor()
