import torch
from torch import nn

from torchvision.ops import roi_align, roi_pool as tv_roi_align, tv_roi_pool

if __name__ == "__main__":
    import os.path as osp
    import sys
    ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    sys.path.insert(0, ROOT_DIR)

from models.utils import proposal_utils as ut


def box_to_pooler_format(boxes):
    """
    Convert boxes into format used in RoI pooler.
    Args:
        boxes(list[Tensor]): one tensor for each image in minibatch

    Return:
        pooler_fmt_boxes(Tensor): tensor of shape ((M1 + M2) x 5), concatenated boxes.
            columns are (batch index, x1, y1, x2, y2). 
    """

    pooler_fmt_boxes = []
    for i, box in enumerate(boxes):
        index = torch.full((box.shape[0], 1), i)
        pooler_fmt_boxes.append(torch.cat((index, box), dim=1))
    
    pooler_fmt_boxes = torch.cat(pooler_fmt_boxes).to(boxes[0].device)

    return pooler_fmt_boxes


def get_levels(boxes, canonical_box_size=224, canonical_level=4):
    """
    Which level to fetch feature map for each roi proposals?

    Args:
        boxes (Tensor): (M1 + M2) x 4 shaped tensors.
    
    Returns:
        levels (Tensor): (M1 + M2) 1-d vector, values in (2, 3, 4, 5)
    """

    areas = ut.compute_area(boxes).to(torch.float32)
    areas = torch.sqrt(areas)/224
    areas = torch.log2(areas) + canonical_level
    areas = torch.floor(areas)
    areas = torch.clamp(areas, min=2, max=5).to(torch.int)

    return areas


def roi_pool_function(input, boxes, output_size, spatial_scale):
    pass


def roi_align_function(input, boxes, output_size, spatial_scale):
    pass


class RoIPoolingLayer(nn.Module):
    def __init__(
        self, 
        output_size,
        spatial_scale
    ):
        super().__init__()


class RoIAligningLayer(nn.Module):
    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio=0,
        aligned=True
    ):
        super().__init__()


class RoIPooler(nn.Module):
    def __init__(
        self,
        output_size = 7,
        scales = (1/4, 1/8, 1/16, 1/32),
        method = "RoIpool"  # RoIPool or RoIAlign
    ):
        super().__init__()
        self.levels = (2, 3, 4, 5)
        self._leveler = get_levels
        self._formatter = box_to_pooler_format

        if method == "RoIPool":
            self.level_poolers = nn.ModuleList(RoIPoolingLayer(output_size, scale) for scale in scales)
        elif method == "RoIAlign":
            self.level_poolers = nn.ModuleList(RoIAligningLayer(output_size, scale) for scale in scales)
        else:
            raise ValueError("unknown pooling method")


    def forward(self, feature_maps, boxes):
        """
        Apply roi pooling or roi aligning.
        
        Args:
            feature_maps (list[Tensor]): (N x C x H2 x W2, ..., N x C x H5 x W5)
            boxes (list[Tensor]): (M1 x 4, M2 x 4), where M1 and M2 are number of proposals for each image in minibatch

        Returns:
            tensor of shape (M1 + M2) x C x (out_size) x (out_size)
            concatenated roi feature map over every batch images.
            each C x (out_size) x (out_size) feature maps are sorted by image order. 
            (i.e. first half is from first image of minibatch, and latter half is from second image.)
        """
        # get feature levels
        box_levels = self._leveler(torch.cat(boxes))

        # into pooler format (image index, x1, y1, x2, y2) and concatenate
        pooler_fmt_boxes = self._formatter(boxes)   # (M1 + M2) x 5

        level_masks = []
        for x in self.levels:
            level_masks.append(box_levels == x)
                
        M = pooler_fmt_boxes.shape[0]
        C = feature_maps[0].shape[1]

        # pooled feature map will be put here
        roi_features = torch.zeros((M, C, self.out_size, self.out_size), 
            dtype=feature_maps[0].dtype, 
            device=feature_maps[0].device)

        for i, pooler in enumerate(self.level_poolers):
            level = self.levels[i]
            feature_map = feature_maps[i]
            level_mask = level_masks[i]
            pooler_fmt_level_boxes = pooler_fmt_boxes[level_mask]
            
            # pool
            pooled_roi_feature_map = pooler(feature_map, pooler_fmt_level_boxes)

            # put into output tensor
            roi_features.index_put_((level_mask, ), pooled_roi_feature_map)
        
        return roi_features















































class RoIPooler(nn.Module):
    def __init__(
        self,
        output_size = 7,
        method = "RoIpool"  # RoIPool or RoIAlign
    ):
        self.out_size = output_size
        self._leveler = get_levels

    def forward(self, feature_maps, boxes):
        """
        Apply roi pooling or roi aligning.
        
        Args:
            feature_maps (list[Tensor]): (N x C x H2 x W2, ..., N x C x H5 x W5)
            boxes (list[Tensor]): (M1 x 4, M2 x 4), where M1 and M2 are number of proposals for each image in minibatch
        
        Returns:
            tensor of shape (M1 + M2) x C x (out_size) x (out_size)
            concatenated roi feature map over every batch images.
        """
        # cat proposals
        proposals = torch.cat(boxes)
        levels = get_levels(proposals)

        level_masks = []
        for x in (2,3,4,5):
            level_masks.append(levels == x)
        
        M = sum([box.shape[0] for box in boxes])
        C = feature_maps[0].shape[1]
        roi_features = torch.zeros((M, C, self.out_size, self.out_size))
        
        for feature_map, level_mask in zip(feature_maps, level_masks):
            # pool
            roi_features = self.pooler(roi_features, feature_map, boxes, level_mask)

        pass











































def roi_pool(roi_features, feature_map, boxes, level_mask):
    """
    RoI pooling function.
    update only boxes corresponding to true values in level_mask.
    Args:
        roi_features (Tensor): features to update, in shape (M x C x out_shape(7) x out_shape(7))
        feature_map (Tensor): feature map of batched images, in shape (N x C x Hi x Wi)
        boxes (Tensor): every proposed boxes in minibatch, in shape (M x 4)
        level_mask (Tensor vector): true values assigned to boxes assigned to this level.
            in shape (M)
    """
    # debug: check if this function only modifies allowed boxes
    for feature in roi_features:
        pass

    return roi_features


def roi_align():
    pass


if __name__ == "__main__":
    box = [
        [100, 100, 120, 120],
        [100, 100, 150, 150],
        [100, 100, 200, 200],
        [100, 100, 300, 300],
        [100, 100, 400, 400],
        [100, 100, 500, 500],
        [100, 100, 600, 600],
        [100, 100, 700, 700],
        [100, 100, 800, 800],
        [100, 100, 900, 900],
        [100, 100, 1000, 1000],
        [100, 100, 1100, 1100],
    ]

    
    box2 = [
        [0, 0, 300, 300],
        [0, 0, 400, 400],
        [0, 0, 500, 500],
        [0, 0, 600, 600],
        [0, 0, 700, 700],
    ]

    box = torch.tensor(box, device=torch.device("cuda:0"))
    box2 = torch.tensor(box2, device=torch.device("cuda:0"))

    boxes = [box, box2]

    proposals = torch.cat(boxes)
    
    levels = get_levels(proposals)
    level_masks = []
    for x in (2,3,4,5):
        level_masks.append(levels == x)

    feature_maps = [torch.ones((2, 3, x, x)) for x in (6,5,4,3,2)]
    M = sum([box.shape[0] for box in boxes])
    C = feature_maps[0].shape[1]
    roi_features = torch.zeros((M, C, 7, 7))

    for feature_map, level_mask in zip(feature_maps, level_masks):
        print(torch.sum(roi_features[level_mask]), "should be 0")
        roi_features = roi_pool(roi_features, feature_map, boxes, level_mask)
        print(torch.sum(roi_features[level_mask]), "should be {}".format(feature_map.shape[2] * feature_map.shape[3]))
    
    print("")

    for feature_map, level_mask in zip(feature_maps, level_masks):
        print(torch.sum(roi_features[level_mask]), "should be {}".format(feature_map.shape[2] * feature_map.shape[3]))


