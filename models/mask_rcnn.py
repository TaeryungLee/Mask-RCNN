import torch
from torch import nn
from torchvision.transforms import ToPILImage
from torch.nn import functional as F
from models.backbone import build_backbone
from models.proposal_generator import build_proposal_generator
from models.heads import build_roi_heads
import numpy as np
from utils.visualizer import vis_tensor, vis_batch, vis_numpy, denormalize_tensor, vis_denorm_tensor_with_bbox, vis_gt_and_prop

class MaskRCNN(nn.Module):
    def __init__(
        self,
        args,
        pixel_mean,
        pixel_std
    ):
        super().__init__()
        self.backbone, self.loaded_layers = build_backbone(args)
        self.proposal_generator = build_proposal_generator(args)
        self.roi_heads = build_roi_heads(args)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        self.args = args

    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def preprocess(self, args, batched_inputs):
        """
        Normalize, pad and batch the input images.
        Must store previous image sizes.
        """
        # vis_numpy(batched_inputs[0]["image"], "vis/" + str(batched_inputs[0]["image_id"]) + "_proc1.jpeg")
        images = [torch.Tensor(x["image"]).to(self.device()) for x in batched_inputs]
        images = [x.permute(2, 0, 1).contiguous() for x in images]
        # images = [torch.reshape(x, (x.shape[-1], x.shape[0], x.shape[1])) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # vis_tensor(images[0], "vis/" + str(batched_inputs[0]["image_id"]) + "_proc2.jpeg")

        # rev = denormalize_tensor(images[0])
        # vis_tensor(rev, "vis/" + str(batched_inputs[0]["image_id"]) + "_denorm.jpeg")




        """
        class ImageList.from_tensors(tensors: List[torch.Tensor], size_divisibility=0, pad_value=0)
        input tensors in format of list[T1, T2, ...], each Ti in shape (C, Hi, Wi).
        size_divisibility = 32 if using FPN.

        Zero paddings are added at right, bottom of image.
        """

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        image_sizes_whole_tensor = torch.tensor(image_sizes)
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if args.model_name == "ResNet-50-FPN":
            stride = self.size_divisibility = 32
        else:
            raise NotImplementedError("such model is not implemented")

        max_size = (max_size + (stride - 1)) // stride * stride
        batch_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)

        i = 0
        # new_batched_imgs = torch.zeros(batch_shape)

        # for img, pad_img in zip(images, new_batched_imgs):
        #     # print(img.shape)
        #     # print(pad_img.shape)
        #     i += 1
        #     # pad_img[:3, :img.shape[1], :img.shape[2]] = img
        #     pad_img[:, :img.shape[-2], :img.shape[-1]] = img
        #     vis_tensor(denormalize_tensor(pad_img), "vis/padded_{}.jpeg".format(i))
        
        # for img in images:
        #     i += 1
        #     # padded = F.pad(input=img, pad=(0, max_size[-1] - img.shape[-1], 0, max_size[-2] - img.shape[-2]))
        #     padded = pad(img, (10, 10))
        #     print(batch_shape)
        #     print(padded.shape)
        #     print(img.shape)
        #     vis_tensor(denormalize_tensor(padded), "vis/denorm_padded_{}.jpeg".format(i))
        #     vis_tensor(denormalize_tensor(img), "vis/denorm_before_padded_{}.jpeg".format(i))
        #     vis_tensor(padded, "vis/padded_{}.jpeg".format(i))
        #     vis_tensor(img, "vis/before_padded_{}.jpeg".format(i))


        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # print(img.shape, "img")
            # print(pad_img.shape, "pad")
            # print(batched_imgs.shape, "batched_imgs")
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        
        annotations = [x["annotations"] for x in batched_inputs]
        
        # print("ann before aligning in model preprocessing")
        # print(annotations)
        

        anno_tensor = align_annotation_size(annotations)
        # print("ann after aligning")
        # print(anno_tensor)
        # print("image ids in preprop")

        image_ids = [x["image_id"] for x in batched_inputs]
        image_ids = torch.tensor(image_ids)
        # print(image_ids)



        return batched_imgs, image_sizes_whole_tensor, anno_tensor, image_ids

    def forward(self, batched_imgs, image_sizes, annotations, image_ids, is_training=True):
        annotations = [annotations[:int(len(annotations)/2)], annotations[int(len(annotations)/2):]]

        anno1 = [x for x in annotations[0] if not (x[0] < 1 and x[1] < 1 and x[2] < 1 and x[3] < 1)]
        anno2 = [x for x in annotations[1] if not (x[0] < 1 and x[1] < 1 and x[2] < 1 and x[3] < 1)]
        annotations = [torch.stack(anno1), torch.stack(anno2)]
        
        backbone_features = self.backbone(batched_imgs)
        roi_proposals, rpn_losses, _, _ = self.proposal_generator(backbone_features, image_sizes, annotations, is_training)
        head_losses, inference, extra = self.roi_heads(batched_imgs, image_sizes, backbone_features, roi_proposals, annotations, is_training=is_training)

        losses = {}

        if is_training:
            losses.update(rpn_losses)
            losses.update(head_losses)
        
        return inference, losses, extra

        



def align_annotation_size(annotations):
    max_len = max([ann.shape[0] for ann in annotations])

    tensors = []
    for ann in annotations:
        if max_len == ann.shape[0]:
            tensors.append(ann)
            continue
        add = torch.zeros((max_len - ann.shape[0], 4))
        tensors.append(torch.cat((ann, add), dim=0))
    # print("tensors in align ftn")
    # print(tensors)
    return torch.cat(tensors, dim=0)








