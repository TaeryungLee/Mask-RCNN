import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from cv2 import rectangle, circle
import cv2


def visualizer(img, bbox, keypoints, i):
    
    lt = (int(bbox[0]), int(bbox[1]))
    rb = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))

    res = add_rectangle(img, (0, 0, 255), lt, rb)

    l = int(len(keypoints) / 3)

    for i in range(l):
        if keypoints[3 * i + 2] == 0:
            continue
        kx = int(keypoints[3*i])
        ky = int(keypoints[3*i + 1])
        kv = int(keypoints[3*i + 2])

        res = add_points(res, (0, 255, 0), (kx, ky))

    return res


def add_rectangle(img, color, lt, rb):
    return rectangle(img, lt, rb, color, 1)

def add_points(img, color, point):
    return circle(img, point, 4, color, -1)


def tensor_to_img(batched_imgs):
    imgs = []
    pixel_mean = torch.tensor([103.53, 116.28, 123.675]).view(-1, 1, 1)
    pixel_std = torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1)
    for tensor_img in batched_imgs:
        img = tensor_img.clone().cpu().detach()
        img = (img * pixel_std) + pixel_mean
        img = torch.reshape(img, (img.shape[1], img.shape[2], img.shape[0]))
        img = img.numpy().astype("uint8")
        
        imgs.append(img)
    
    return imgs

def vis_batch(batched_imgs, image_sizes, annotations, image_ids):
    imgs = tensor_to_img(batched_imgs)

    for img, imid in zip(imgs, image_ids):
        # print(imid)
        # print(img)
        # print(img.shape)
        vis_numpy(img, "vis/" + str(int(imid)) + "_after2" + ".jpeg")

def denormalize_tensor(tensor_image):
    # pixel_mean = torch.tensor([103.53, 116.28, 123.675]).view(-1, 1, 1)
    # pixel_std = torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1)
    # img = tensor_image.clone().cpu().detach()
    # img = (img * pixel_std) + pixel_mean

    img = tensor_image.clone().cpu().detach()

    img[0] += 103.53
    img[1] += 116.28
    img[2] += 123.675
    return img

def vis_tensor(tensor_image, out_name):
    # tensor_image dim (c * h * w)
    img = tensor_image.clone().cpu().detach()
    img = img.permute(1, 2, 0)
    # img = torch.reshape(img, (img.shape[1], img.shape[2], img.shape[0]))
    img = img.numpy().astype("uint8")

    vis_numpy(img, out_name)

def vis_numpy(np_image, out_name):
    # np_image dim (h * w * c)
    image = Image.fromarray(np_image)
    image.save(out_name)

    # cv2.imwrite(out_name, np_image)

def vis_tensor_with_bbox(tensor_image, bboxes, box_format, out_name):
    img = tensor_image.clone().cpu().detach()
    img = img.permute(1, 2, 0).contiguous()
    img = img.numpy().astype("uint8")
    # img = Image.fromarray(img)
    img = img.copy()
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    # print(type(img))
    for bbox in bboxes:
        if bbox[0] < 1 and bbox[1] < 1 and bbox[2] < 1 and bbox[3] < 1:
            continue

        if box_format == "bbox":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))
        elif box_format == "anchor":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[2]), int(bbox[3]))
        else:
            raise TypeError("unknown box format in vis")
        # print(bbox, lt, rb)
        img = add_rectangle(img, (0, 0, 255), lt, rb)
    
    cv2.imwrite(out_name, img)

def vis_denorm_tensor_with_bbox(tensor_image, bboxes, box_format, out_name):
    img = denormalize_tensor(tensor_image)
    vis_tensor_with_bbox(img, bboxes, box_format, out_name)

def vis_gt_and_prop(tensor_image, gtboxes, propboxes, gtformat, propformat, out_name):
    # Proposal as blue, gt as red
    img = denormalize_tensor(tensor_image)

    img = img.clone().cpu().detach()
    img = img.permute(1, 2, 0).contiguous()
    img = img.numpy().astype("uint8")
    # img = Image.fromarray(img)
    img = img.copy()
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    # print(type(img))
    for bbox in propboxes:
        if bbox[0] < 1 and bbox[1] < 1 and bbox[2] < 1 and bbox[3] < 1:
            continue

        if propformat == "bbox":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))
        elif propformat == "anchor":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[2]), int(bbox[3]))

        img = add_rectangle(img, (255, 0, 0), lt, rb)

    for bbox in gtboxes:
        if bbox[0] < 1 and bbox[1] < 1 and bbox[2] < 1 and bbox[3] < 1:
            continue

        if gtformat == "bbox":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))
        elif gtformat == "anchor":
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[2]), int(bbox[3]))
        img = add_rectangle(img, (0, 0, 255), lt, rb)
    
    cv2.imwrite(out_name, img)
