import numpy as np

class AugInput():
    """
    This class holds input informations such as **image, bounding box, segmentation, keypoints**.
    """
    def __init__(self,
        image,
        annotations,
        use_seg=False,
        use_key=True):

        self.image = image
        self.annotations = annotations
        