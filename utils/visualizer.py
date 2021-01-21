import PIL
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    return rectangle(img, lt, rb, color, 2)

def add_points(img, color, point):
    return circle(img, point, 4, color, -1)