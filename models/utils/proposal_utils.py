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


def compute_iou():
    pass


def compute_overlaps():
    pass


def find_topn_proposals():
    pass


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
