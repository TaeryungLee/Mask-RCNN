import os
import torch
import numpy as np
import config
import copy
import cv2
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
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
                # assert ann["category_id"] in target_class

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

        # Test visualization code
        # vis = img.copy()
        # vis[:, :, [0, 2]] = vis[:, :, [2, 0]]
        # i = 0
        # for obj in dataset_dict["annotations"]:
        #     vis = visualizer(vis, obj["bbox"], obj["keypoints"], i)
        #     i += 1
        # cv2.imwrite("visuailization_pre.jpeg", vis)

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
        
        # Finally, I have to convert annotations to instances. (To be implemented)
        # raise NotImplementedError("Convert annotations to instances not implemented")

        return ret

def test():
    from main.train import parse_args
    args = parse_args()

    data = MSCOCO(args, "/media/thanos_hdd0/taeryunglee/detectron2/coco", train=False)

    res = data.__getitem__(2133)

    im = Image.fromarray(res["image"])
    im.save("after.jpeg")

    i = 0
    vis = res["image"].copy()
    vis[:, :, [0, 2]] = vis[:, :, [2, 0]]

    for obj in res["annotations"]:
        vis = visualizer(vis, obj["bbox"], obj["keypoints"], i)
        i += 1
    cv2.imwrite("visuailization.jpeg", vis)
    

if __name__ == "__main__":
    test()
