import torch
from torch import nn
from torch.nn import functional as F
from models.backbone import build_backbone
from models.proposal_generator import build_proposal_generator
from models.roi_heads import build_roi_heads
import numpy as np


class MaskRCNN(nn.Module):
    def __init__(
        self,
        args,
        pixel_mean,
        pixel_std
    ):
        super().__init__()
        self.backbone = build_backbone(args)
        self.proposal_generator = build_proposal_generator(args)
        self.roi_heads = build_roi_heads(args)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_mean).view(-1, 1, 1))

    def device(self):
        return self.pixel_mean.device

    
    def preprocess(self, args, batched_inputs):
        """
        Normalize, pad and batch the input images.
        Must store previous image sizes.
        """
        images = [torch.Tensor(x["image"]).to(self.device()) for x in batched_inputs]
        images = [torch.reshape(x, (x.shape[-1], x.shape[0], x.shape[1])) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        """
        class ImageList.from_tensors(tensors: List[torch.Tensor], size_divisibility=0, pad_value=0)
        input tensors in format of list[T1, T2, ...], each Ti in shape (C, Hi, Wi).
        size_divisibility = 32 if using FPN.

        Zero paddings are added at right, bottom of image.
        """

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if args.model_name == "ResNet-50-FPN":
            stride = self.size_divisibility = 32
        else:
            raise NotImplementedError("such model is not implemented")

        max_size = (max_size + (stride - 1)) // stride * stride
        batch_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # print(img.shape, "img")
            # print(pad_img.shape, "pad")
            # print(batched_imgs.shape, "batched_imgs")
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        
        return batched_imgs, image_sizes









