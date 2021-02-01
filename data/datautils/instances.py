import itertools
from typing import Any, Dict, List, Tuple, Union
import torch



def annotations_to_instances(annos, image_size):
    pass


class Instances:
    def __init__(self, image_size, **kwargs):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)
    
    
