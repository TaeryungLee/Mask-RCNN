import numpy as np
from PIL import Image


def load_image(file_name, format="RGB"):
    """
    Read an image into the given format.
    """
    image = Image.open(file_name).convert(format)

    return file_name

def PIL_to_numpy(image, format=None):
    if format:
        raise NotImplementedError("pil to numpy format not implemented")
    image = np.asarray(image)
    return image

def check_image_size(image, height, width):
    imsize = get_image_size(image)
    return (height == imsize[0]) and (width == imsize[1])
    

def get_image_size(image):
    # H * W
    return (image.shape[0], image.shape[1])


