import os
import torch
import numpy as np
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image

class MSCOCO(Dataset):
    def __init__(self, data_dir, target_class=[1], train=True, add_ann=[]):
        """
        Check directory

        data should be in directory like this:
        data_dir/
            annotations/
                instances_{train,val}2017.json
                person_keypoints_{train,val}2017.json
            {train,val}2017/
                # image files that are mentioned in the corresponding json
        """

        assert os.path.isdir(os.path.join(data_dir, "annotations"))
        assert os.path.isdir(os.path.join(data_dir, "train2017"))
        assert os.path.isdir(os.path.join(data_dir, "val2017"))

        assert os.path.isfile(os.path.join(data_dir, "annotations", "person_keypoints_train2017.json"))
        assert os.path.isfile(os.path.join(data_dir, "annotations", "person_keypoints_val2017.json"))

        """
        look at coco.py/load_coco_json("person_keypoints_train2017.json", imageroot, "keypoints_coco_2017_train")
        """

        if train:
            self._ann_path = os.path.join(data_dir, "annotations", "person_keypoints_train2017.json")
            self._data_path = os.path.join(data_dir, "train2017")
        else:
            self._ann_path = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")
            self._data_path = os.path.join(data_dir, "val2017")
        
        self.coco = COCO(self._ann_path)
        self.whole_ids = self.coco.getImgIds()
        self.person_ids = self.coco.getImgIds(catIds=target_class)

        self.imgs = self.coco.loadImgs(self.person_ids)
        self.anns = [self.coco.imgToAnns[idx] for idx in self.person_ids]

        self.imgs_anns = list(zip(self.imgs, self.anns))

        self.out_ann = ["num_keypoints", "bbox", "iscrowd"] + add_ann
        self.out_ann = list(set(self.out_ann))

        self.dataset_dict = []

        for (img_dict, ann_dict_list) in self.imgs_anns:
            record = {}
            record["file_name"] = os.path.join(self._data_path, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for ann in ann_dict_list:
                assert ann["image_id"] == image_id
                obj = {key: ann[key] for key in self.out_ann if key in ann}
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
                self.dataset_dict.append(record)

    def __len__(self):
        return len(self.dataset_dict)
    
    def __getitem__(self, idx):
        record = self.dataset_dict[idx]

        filename = record["file_name"]
        target = record["annotations"]

        img = Image.open(filename).convert("RGB")

        return img, target

def test():
    data = MSCOCO("/media/thanos_hdd0/taeryunglee/detectron2/coco", train=False)
    print(len(data))

    img, target = data.__getitem__(2)

    pprint(target)
    img.save("tmp.jpeg")

if __name__ == "__main__":
    test()