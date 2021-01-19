import numpy as np
from abc import abstractmethod, ABCMeta
from data.datautils.augmentations import AugInput


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
    def _apply_coords(self, coords: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, aug_input: AugInput):
        pass


class ResizeShortestEdge(Transform):
    """
    Implementation of ResizeShortestEdge augmentation of detectron2.
    It contains simplified method to do the same job as original.
    """
    def __init__(self):
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
        sample_style (str, "range" or "choice")
        """
        pass

    def _resize_image(self, img):
        pass

    def _resize_coords(self, coords):
        pass

    def _apply_image(self, img):
        pass

    def _apply_coords(self, coords):
        pass
    
    def __call__(self, aug_input):
        pass

class RandomFlip(Transform):
    """
    Implementation of RandomFlip augmentation of detectron2.
    It contains simplified method to do the same job as original.
    """

    def __init__(self):
        pass

    def _apply_image(self, img):
        pass

    def _apply_coords(self, coords):
        pass
    
    def __call__(self, aug_input):
        pass