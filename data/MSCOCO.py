import os
import torch
import numpy as np
import config
import copy
import cv2
import json
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from data.datautils import image as imutil
from data.datautils import augmentations as aug
from data.datautils import transforms as T
from utils.visualizer import visualizer


class MSCOCO(Dataset):
    def _build_dataset_dict(self, ann_path, data_path, target_class, add_ann):
        dataset_dicts = []

        coco = COCO(ann_path)
        self.person_ids = coco.getImgIds(catIds=target_class)

        imgs = coco.loadImgs(self.person_ids)
        anns = [coco.imgToAnns[idx] for idx in self.person_ids]

        imgs_anns = list(zip(imgs, anns))

        out_ann = ["bbox", "category_id"] + add_ann
        out_ann = list(set(out_ann))

        for (img_dict, ann_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(data_path, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for ann in ann_dict_list:
                if ann["category_id"] in target_class:
                    assert ann["image_id"] == image_id
                    obj = {key: ann[key] for key in out_ann if key in ann}
                    assert ann["category_id"] in target_class

                    objs.append(obj)
            
            record["annotations"] = objs

            dataset_dicts.append(record)
        return dataset_dicts
    
    def _build_transforms(self, args, transforms):
        """
        This function returns a list of initialized Transform class objects. (list[Transform])
        Argument :args: holds informations needed to initialize Transform objects.
        Argument :transforms: list[str]: holds names of transforms to be initialized. 
        """
        transform_list = []

        for t in transforms:
            if t == "ResizeShortestEdge":
                # initialize ResizeShortestEdge and append
                transform_list.append(T.ResizeShortestEdge(args.min_size.split(), args.max_size))

            elif t == "RandomFlip":
                # initialize RandomFlip and append
                transform_list.append(T.RandomFlip(0.5, "h"))
        return transform_list

    def __init__(self, 
        args, data_dir, target_class=[1], train=True, add_ann=[], 
        transforms=["ResizeShortestEdge", "RandomFlip"]):
        assert os.path.isdir(os.path.join(data_dir, "annotations")), "no annotation dir"
        assert os.path.isdir(os.path.join(data_dir, "train2017")), "no train dir"
        assert os.path.isdir(os.path.join(data_dir, "val2017")), "no validation dir"
        assert os.path.isfile(os.path.join(data_dir, "annotations", "instances_train2017.json")), "no train annotation file"
        assert os.path.isfile(os.path.join(data_dir, "annotations", "instances_val2017.json")), "no validation annotation file"

        self.train = train

        if train:
            self._ann_path = os.path.join(data_dir, "annotations", "instances_train2017.json")
            self._data_path = os.path.join(data_dir, "train2017")
        else:
            self._ann_path = os.path.join(data_dir, "annotations", "instances_val2017.json")
            self._data_path = os.path.join(data_dir, "val2017")

        self.dataset_dicts = self._build_dataset_dict(self._ann_path, self._data_path, target_class, add_ann)

        # debug: 1/100 training data, 1/10 validation data
        # if train:
        #     self.dataset_dicts = self.dataset_dicts[::100]
        # else:
        #     self.dataset_dicts = self.dataset_dicts[::10]

        self.transforms = self._build_transforms(args, transforms)

    def __len__(self):
        return len(self.dataset_dicts)
    
    def __getitem__(self, idx):
        # First, get deep copy of dataset_dict
        dataset_dict = copy.deepcopy(self.dataset_dicts[idx])

        img = imutil.load_image(dataset_dict["file_name"])
        img = imutil.PIL_to_numpy(img)

        assert imutil.check_image_size(img, dataset_dict["height"], dataset_dict["width"]), "image size of {} does not match!".format(dataset_dict["file_name"])

        img_shape = imutil.get_image_size(img) # H * W
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2,0,1)))

        # Build AugInput
        aug_input = aug.AugInput(img, dataset_dict["annotations"])
        
        for transform in self.transforms:
            aug_input = transform(aug_input)

        dataset_dict["image"] = aug_input.image.copy()

        bboxes = [x["bbox"] for x in aug_input.annotations]
        dataset_dict["annotations"] = torch.tensor(bboxes)

        ret = {
            "image": dataset_dict["image"],
            "annotations": torch.tensor(bboxes)
        }
        
        return dataset_dict


class COCO_custom_evaluator():
    def __init__(
        self,
        data_dir,
        home_dir,
        target_class=[1]    
    ):
        self.home_dir = home_dir
        self._ann_type = 'bbox'
        self.img_ids = None
    
    def evaluate(self, gt, list_dict):
        # dump gt
        gt_file = os.path.join(self.home_dir, "gt_tmp.json")
        with open(gt_file, "w") as f:
            json.dump(gt, f)

        # dump prediction
        eval_file = os.path.join(self.home_dir, "eval_tmp.json")
        with open(eval_file, "w") as f:
            json.dump(list_dict, f)
        
        self.coco_gt = COCO(gt_file)
        coco_dt = self.coco_gt.loadRes(eval_file)

        if self.img_ids is None:
            img_ids = [x["image_id"] for x in list_dict]
            img_ids = sorted(list(set(img_ids)))
            self.img_ids = img_ids

        coco_eval = COCOeval(self.coco_gt, coco_dt, self._ann_type)
        coco_eval.params.catIds = [1]
        coco_eval.params.imgIds = self.img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        """
        stat fields
        Average Precision @ IoU=0.50:0.95 | area=   all |
        Average Precision @ IoU=0.50      | area=   all |
        Average Precision @ IoU=0.75      | area=   all |
        Average Precision @ IoU=0.50:0.95 | area= small |
        Average Precision @ IoU=0.50:0.95 | area=medium |
        Average Precision @ IoU=0.50:0.95 | area= large |
        Average Recall    @ IoU=0.50:0.95 | area=   all |
        Average Recall    @ IoU=0.50:0.95 | area=   all |
        Average Recall    @ IoU=0.50:0.95 | area=   all |
        Average Recall    @ IoU=0.50:0.95 | area= small |
        Average Recall    @ IoU=0.50:0.95 | area=medium |
        Average Recall    @ IoU=0.50:0.95 | area= large |

        대표 ap, ar: 0, 6
        """

        os.remove(eval_file)
        os.remove(gt_file)
        return coco_eval.stats, coco_eval.strings
