import torch
from torch import nn
from torchvision.ops import roi_align as tv_roi_align
from torchvision.ops import roi_pool as tv_roi_pool
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
        index = torch.full((box.shape[0], 1), i, device=boxes[0].device)
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


class RoIPoolingLayer(nn.Module):
    def __init__(
        self, 
        output_size,
        spatial_scale
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, input, boxes):
        """
        Apply torchvision.roi_pool

        Args:
            input (Tensor): shape (N x C x H x W)
            boxes (Tensor): boxes in pooling format (image_index, x1, y1, x2, y2), shape (M x 5)

        Returns:
            output (Tensor): pooled roi feature map, shape (M x C x out_size x out_size)
        """
        output = tv_roi_pool(input, boxes, self.output_size, self.spatial_scale)
        return output

class RoIAligningLayer(nn.Module):
    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio=0,
        aligned=True
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, boxes):
        """
        Apply torchvision.roi_align

        Args:
            input (Tensor): shape (N x C x H x W)
            boxes (Tensor): boxes in pooling format (image_index, x1, y1, x2, y2), shape (M x 5)

        Returns:
            output (Tensor): pooled roi feature map, shape (M x C x out_size x out_size)
        """
        return tv_roi_align(input, boxes, (self.output_size, self.output_size), self.spatial_scale, self.sampling_ratio, self.aligned)


class RoIPooler(nn.Module):
    def __init__(
        self,
        output_size=7,
        scales=(1/4, 1/8, 1/16, 1/32),
        method="RoIpool"  # RoIPool or RoIAlign
    ):
        super().__init__()
        self.levels = (2, 3, 4, 5)
        self._leveler = get_levels
        self._formatter = box_to_pooler_format
        self.out_size = output_size

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
