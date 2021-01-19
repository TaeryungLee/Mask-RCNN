import os
import torch
import numpy as np
import config
import copy
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
from data.datautils import image as imutil
from data.datautils import augmentations as aug
from data.datautils import transforms as T

class MSCOCO(Dataset):
    def _build_dataset_dict(self, ann_path, data_path, target_class, add_ann):
        dataset_dicts = []

        coco = COCO(ann_path)
        self.person_ids = coco.getImgIds(catIds=target_class)

        imgs = coco.loadImgs(self.person_ids)
        anns = [coco.imgToAnns[idx] for idx in self.person_ids]

        imgs_anns = list(zip(imgs, anns))

        out_ann = ["num_keypoints", "bbox", "iscrowd"] + add_ann
        out_ann = list(set(out_ann))

        for (img_dict, ann_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(data_path, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for ann in ann_dict_list:
                assert ann["image_id"] == image_id
                obj = {key: ann[key] for key in out_ann if key in ann}
                assert ann["category_id"] in target_class

                keypts = ann.get("keypoints", None)
                if keypts:
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            keypts[idx] = v + 0.5
                    obj["keypoints"] = keypts
                objs.append(obj)
            
            record["annotations"] = objs

            vis_keypts_in_img = sum(
                (np.array(ann["keypoints"][2::3]) > 0).sum()
                for ann in objs
                if "keypoints" in ann)
            if vis_keypts_in_img > 0:
                dataset_dicts.append(record)
        return dataset_dicts
    
    def _build_transforms(self, args, transforms):
        """
        This function returns a list of initialized Transform class objects. (list[Transform])
        Argument :args: holds informations needed to initialize Transform objects.
        Argument :transforms: list[str]: holds names of transforms to be initialized. 
        """
        pass

    def __init__(self, 
        args, data_dir, target_class=[1], train=True, add_ann=[], 
        transforms=["ResizeShortestEdge", "RandomFlip"]):
        assert os.path.isdir(os.path.join(data_dir, "annotations")), "no annotation dir"
        assert os.path.isdir(os.path.join(data_dir, "train2017")), "no train dir"
        assert os.path.isdir(os.path.join(data_dir, "val2017")), "no validation dir"
        assert os.path.isfile(os.path.join(data_dir, "annotations", "person_keypoints_train2017.json")), "no train annotation file"
        assert os.path.isfile(os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")), "no validation annotation file"

        self.train = train

        if train:
            self._ann_path = os.path.join(data_dir, "annotations", "person_keypoints_train2017.json")
            self._data_path = os.path.join(data_dir, "train2017")
        else:
            self._ann_path = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")
            self._data_path = os.path.join(data_dir, "val2017")

        self.dataset_dicts = self._build_dataset_dict(self._ann_path, self._data_path, target_class, add_ann)
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
        dataset_dict["image"] = torch.as_tnesor(np.ascontiguousarray(img.transpose(2,0,1)))

        # Build AugInput
        aug_input = aug.AugInput(img, dataset_dict["annotations"])

        # We have built transform list in __init__.
        # So we just need to put aug_input into given transforms sequentially.

        for transform in self.transforms:
            aug_input = transform(aug_input)
        
        # Extract image and annotation data from aug_input
        # In fact, I have to make every transforms work in place, 
        # so I don't have to extract results from aug_input actually. 
        # Anyways, I will do extraction.

        dataset_dict["image"] = aug_input.image
        dataset_dict["annotations"] = aug_input.annotations
        
        # Finally, I have to convert annotations to instances. (To be implemented)
        raise NotImplementedError("Convert annotations to instances not implemented")


        return dataset_dict

def test():
    data = MSCOCO("/media/thanos_hdd0/taeryunglee/detectron2/coco", train=True)
    print(len(data))

    img, target = data.__getitem__(2)

    # pprint(target)
    img.save("tmp.jpeg")

if __name__ == "__main__":
    test()