import os
import sys
import config
import torch
import argparse
import pickle
from torch import nn

from data.loader import build_dataloader
from data.MSCOCO import COCO_custom_evaluator
from models.mask_rcnn import MaskRCNN
from utils.visualizer import vis_denorm_tensor_with_bbox, vis_gt_and_prop
from utils.dir import mkdir
from utils.logger import ColorLogger as Logger
from utils.timer import sec2minhrs

from main.train import load_my_rpn, box_to_eval_form, box_to_gt_form

tested_params = [
    "rpn_nms_threshs",
    "rpn_nms_topk_trains",
    "rpn_nms_topk_tests",
    "rpn_nms_topk_posts",
    "roi_test_score_threshs",
    "roi_nms_threshs",
    "roi_nms_topk_posts"
]

class Tester():
    def __init__(self, args):
        self.args = args
        self.logger = Logger(self.args.home)

        self.test_loader = self._create_test_dataloader(args)
        self.evaluator = COCO_custom_evaluator(args.data_dir, args.home)
    

    def _create_model(self, args, model_name):
        self.logger.info("creating model...")

        pixel_mean = args.pixel_mean.split()
        pixel_mean = [float(x) for x in pixel_mean]
        pixel_std = args.pixel_std.split()
        pixel_std = [float(x) for x in pixel_std]

        if model_name == "ResNet-50-FPN":
            model = MaskRCNN(args, pixel_mean, pixel_std)
        return model
    

    def _create_val_dataloader(self, args):
        return build_dataloader(args, train=False)


    def test(self):
        models = [
            "./output/test09/best.pkl",
        ]
        
        rpn_nms_threshs = [
            0.6, 0.7, 0.8
        ]

        rpn_nms_topk_trains = [
            2000
        ]

        rpn_nms_topk_tests = [
            500, 1000
        ]

        rpn_nms_topk_posts = [
            500, 1000
        ]

        roi_test_score_threshs = [
            0.4, 0.5, 0.6
        ]

        roi_nms_threshs = [
            0.4, 0.5, 0.6
        ]

        roi_nms_topk_posts = [
            100
        ]

        results = []
        for model_path in models:

            for grid in zip(
                rpn_nms_threshs,
                rpn_nms_topk_trains,
                rpn_nms_topk_tests,
                rpn_nms_topk_posts,
                roi_test_score_threshs,
                roi_nms_threshs,
                roi_nms_topk_posts
            ):
                self.args.rpn_nms_threshs = grid[0]
                self.args.rpn_nms_topk_trains = grid[1]
                self.args.rpn_nms_topk_tests = grid[2]
                self.args.rpn_nms_topk_posts = grid[3]
                self.args.roi_test_score_threshs = grid[4]
                self.args.roi_nms_threshs = grid[5]
                self.args.roi_nms_topk_posts = grid[6]

                # create grid model                
                self.model = self._create_model(args, model_name=args.model_name)
                load_my_rpn(self.model, model_path)
                self.model = nn.DataParallel(self.model)
                self.model.cuda()

                # do test
                result_boxes = []
                gt_boxes = []
                img_ids = []
                for i, data in enumerate(self.test_loader):
                    batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
                
                    with torch.no_grad():
                        output, _, _ = self.model(batched_imgs, image_sizes, annotations, image_ids, is_training=False)
                    
                    ret_boxes, ret_scores, _, num_inferences = output
                    num_inferences = num_inferences.tolist()
                    ret_boxes = ret_boxes.split(num_inferences)
                    ret_scores = ret_scores.split(num_inferences)

                    for image_id, boxes_per_image, scores_per_image in zip(image_ids, ret_boxes, ret_scores):
                        for box, score in zip(boxes_per_image.tolist(), scores_per_image.tolist()):
                            bbox = {
                                "image_id": int(image_id),
                                "category_id": 1,
                                "bbox": box_to_eval_form(box),
                                "score": score
                            }
                            result_boxes.append(bbox)
                    for image_id, gt_boxes_per_image in zip(image_ids, [x["annotations"] for x in data]):
                        for box in gt_boxes_per_image.tolist():
                            _id = _id + 1
                            # box to x, y, width, height form
                            gt_boxes.append(box_to_gt_form(box, image_id, _id))
                            img_ids.append({"id": int(image_id)})
                
                gt = {"annotations": gt_boxes, "images": img_ids, "categories": [{"supercategory": "person", "id": 1, "name": "person"}]}
                stats, strings = self.evaluator.evaluate(gt, result_boxes)
                mAP, mAR = stats[0], stats[8]
                results.append(grid, strings, stats, mAP, mAR)

                self.logger.info("")
                self.logger.info("All settings tested:")
                for k in tested_params:
                    self.logger.debug("{}: {}".format(k, vars(self.args)[k]))
                self.logger.info("Result:")
                for string in strings:
                    self.logger.info(string)
                self.logger.info("mAP: {}, mAR: {}".format(mAP, mAR))
        
        self.logger.info("")
        self.logger.info("==========================================================================================")
        self.logger.info("")

        # by mAP
        results.sort(reverse=True, key=lambda x : x[3])
        self.logger.info("top 10 results by mAP")
        for i, result in enumerate(results[:10]):
            self.logger.info("Used settings:")
            for param, name in zip(result[0], tested_params):
                self.logger.info("{}: {}".format(name, param))
            for string in result[1]:
                self.logger.info(string)
            self.logger.info("mAP: {}, mAR: {}".format(result[3], result[4]))

        # by mAR
        self.logger.info("top 10 results by mAR")
        results.sort(reverse=True, key=lambda x : x[4])
        for i, result in enumerate(results[:10]):
            self.logger.info("Used settings:")
            for param, name in zip(result[0], tested_params):
                self.logger.info("{}: {}".format(name, param))
            for string in result[1]:
                self.logger.info(string)
            self.logger.info("mAP: {}, mAR: {}".format(result[3], result[4]))


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--is_train', type=bool, default=False)
    
    _parser.add_argument('--num_gpus', type=int, default=2)
    _parser.add_argument('--gpu_ids', type=str, 
                         help="Use comma between ids")
    _parser.add_argument('--home', type=str, default="./output/test_final")
    _parser.add_argument('--num_workers', type=int, default=4)

    _parser.add_argument('--model_name', type=str, default="ResNet-50-FPN")

    _parser.add_argument('--image_per_batch', type=int, default=2)
    _parser.add_argument('--batch_size', type=int, default=4)
    
    _parser.add_argument('--min_size', type=str, default='640 672 704 736 768 800')
    _parser.add_argument('--max_size', type=int, default=1333)

    _parser.add_argument('--data_dir', type=str, 
                         default="/media/thanos_hdd0/taeryunglee/detectron2/coco")

    _parser.add_argument('--pixel_mean', type=str, default='103.53 116.28 123.675')
    _parser.add_argument('--pixel_std', type=str, default='1.0 1.0 1.0')
    _parser.add_argument('--fpn_out_chan', type=int, default=256)

    _parser.add_argument('--load_pretrained_resnet', type=bool, default=False)
    _parser.add_argument('--pretrained_resnet', type=str, default="pretrained/R-50.pkl")

    _parser.add_argument('--rpn_batch_size', type=int, default=256)
    _parser.add_argument('--roi_batch_size', type=int, default=256)

    # Hyperparameters to test
    _parser.add_argument('--rpn_nms_thresh', type=float, default=0.7)
    _parser.add_argument('--rpn_nms_topk_train', type=int, default=2000)
    _parser.add_argument('--rpn_nms_topk_test', type=int, default=1000)
    _parser.add_argument('--rpn_nms_topk_post', type=int, default=1000)

    
    _parser.add_argument('--roi_test_score_thresh', type=float, default=0.5)
    _parser.add_argument('--roi_nms_thresh', type=float, default=0.4)
    _parser.add_argument('--roi_nms_topk_post', type=int, default=25)
    
    args = _parser.parse_args()
    return args



def main():
    print(sys.argv)
    args = parse_args()
    
    tester = Tester(args)
    Tester.test()

if __name__ == "__main__":
    main()