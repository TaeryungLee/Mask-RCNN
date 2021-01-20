import numpy as np
from abc import abstractmethod, ABCMeta
from data.datautils.augmentations import AugInput
from PIL import Image
import random


class Transform(metaclass=ABCMeta):
    """
    Base class for deterministic transform function.
    Transforms for both images and coordinates should be always implemented.

    Detectron2 aligned transforms on annotation with image by returning reusable :Transform: object.
    I want to simplify this procedure to apply transform on both image and annotations at the same time
    by calling this :Transform: object.
    Method __call__ should include decision on precise deterministic behavior on both image and annotations,
    and should call _apply_image, _apply_coords methods.
    It should return transform-applied aug_input object, which contains transform-applied image and 
    transform-applied coords.
    """
    def __init__(self):
        pass

    @abstractmethod
    def _apply_image(self, img: np.ndarray):
        pass

    @abstractmethod
    def _apply_bbox(self, bbox):
        pass

    @abstractmethod
    def _apply_keypoints(self, keypoints):
        pass

    @abstractmethod
    def __call__(self, aug_input: AugInput):
        pass


class ResizeShortestEdge(Transform):
    """
    Implementation of ResizeShortestEdge augmentation of detectron2.
    It contains simplified method to do the same job as original.
    """
    def __init__(self, min_size, max_size):
        """
        (미리 MSCOCO.__init__에서 룰에 맞는(하이퍼파라미터에 맞는) transform 객체를 생성시켜 놓고,
        이후에 하나씩 이미지데이터를 통과시키는 형식임.)
        
        detectron2에서는 get_transform이 transform 생성. 이미지를 하나 받아서 w, h를 크기정보 삼아서
        나머지 annotation을 수정할 transform을 생성하는 것.

        여기서는 __call__에서 그걸 한군데로 묶어서 시행할 예정. 저기서는 transform을 내보냄으로써 이미지를 처리하는거랑
        annotation 처리하는거 데이터를 일치시켰다면 여기서는 한 번에 처리함으로써 일치시킬 것.

        Hyperparameter required:
        min_size (list[int])
        max_size (int)
        """
        self.min_size = min_size
        self.max_size = max_size

    def _apply_image(self, image, h, w, newh, neww):
        assert image.shape[:2] == (h, w)
        assert len(image.shape) <= 4

        interp_method = Image.BILINEAR

        if image.dtype == np.uint8:
            if len(image.shape) > 2 and image.shape[2] == 1:
                pil_image = Image.fromarray(image[:, :, 0])
            else:
                pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((neww, newh), interp_method)
            ret = np.asarray(pil_image)
            if len(image.shape) > 2 and image.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        
        else:
            raise NotImplementedError("image dtype not uint8")

        return ret

    def _apply_bbox(self, bbox, h, w, newh, neww):
        r = neww * 1.0 / w
        newbbox = [round(x * r, 2) for x in bbox]
        return newbbox

    def _apply_keypoints(self, keypoints, h, w, newh, neww):
        newkeypoints = []
        r = neww * 1.0 / w

        for i, kpt in enumerate(keypoints):
            if i % 3 == 2:
                newkeypoints.append(kpt)
            else:
                newkeypoints.append(round((kpt * r), 2))
        return newkeypoints

    def _get_new_size(self, h, w):
        size = int(np.random.choice(self.min_size))
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww

    def __call__(self, aug_input):
        image, anno = aug_input.image, aug_input.annotations
        h, w = image.shape[:2]
        newh, neww = self._get_new_size(h, w)
        image = self._apply_image(image, h, w, newh, neww)
        aug_input.image = image

        for obj in anno:
            obj["keypoints"] = self._apply_keypoints(obj["keypoints"], h, w, newh, neww)
            # to be implemented: bbox
            obj["bbox"] = self._apply_bbox(obj["bbox"], h, w, newh, neww)
        return aug_input


class RandomFlip(Transform):
    """
    Implementation of RandomFlip augmentation of detectron2.
    It contains simplified method to do the same job as original.
    """

    def __init__(self, prob, direction):
        """
        Hyperparameter required:
        prob (int): probability of flip.
        direction (str, "h" or "v"): horizontal or vertical.
        """
        self.prob = prob
        self.horizontal = (direction == 'h')

    def _apply_image(self, img):
        if self.horizontal:
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=2)

    def _apply_bbox(self, bbox, h, w):
        return [w-bbox[0]-bbox[2], bbox[1], bbox[2], bbox[3]]

    def _apply_keypoints(self, keypoints, h, w):
        newkeypoints = []

        for i, kpt in enumerate(keypoints):
            if i % 3 == 0:
                newkeypoints.append(w-kpt)
            else:
                newkeypoints.append(kpt)
        return newkeypoints
    
    def __call__(self, aug_input):
        if random.random() < self.prob:
            return aug_input
        
        image, anno = aug_input.image, aug_input.annotations
        h, w = image.shape[:2]
        image = self._apply_image(image)
        aug_input.image = image

        for obj in anno:
            obj["keypoints"] = self._apply_keypoints(obj["keypoints"], h, w)
            # to be implemented: bbox
            obj["bbox"] = self._apply_bbox(obj["bbox"], h, w)
        return aug_input