import logging
import os
import sys
import config
import abc
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.optim import SGD, Adam
from torch import nn
import numpy as np
import argparse
from data.loader import build_dataloader
from data.MSCOCO import COCO_custom_evaluator
from utils.default import DefaultTrainer
from models.mask_rcnn import MaskRCNN
from models.nets.resnet import ResNet
from torchvision.models import resnet50
from utils import visualizer as vis
from PIL import Image
from utils.logger import colorLogger as Logger
from models.utils.proposal_utils import find_top_match_proposals, remove_zero_gt
from utils.visualizer import vis_denorm_tensor_with_bbox, vis_gt_and_prop
from utils.dir import mkdir
import pickle
from utils.timer import sec2minhrs

# Hyperparameters
hyperparams = ['max_iter', 'lr_steps',  'lr', 'rpn_pos_weight', 'roi_pos_weight',
    'rpn_nms_thresh',  'rpn_nms_topk_train',  'rpn_nms_topk_test', 'rpn_nms_topk_post', 'rpn_batch_size',
    'roi_batch_size', 'roi_test_score_thresh', 'roi_nms_thresh', 'roi_nms_topk_post', 'freeze_backbone',
    'freeze_rpn', 'load', 'load_name'
]

class Trainer(DefaultTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.args = args
        self.logger = Logger(self.args.home)
        self.model = self._create_model(args, model_name=args.model_name)

        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.optimizer = self._get_optimizer(model=self.model, args=args)

        self.lr_scheduler = self._get_lr_scheduler(
            lr=args.lr, lr_scheduler=args.lr_scheduler,
            start_iter=args.start_iter, max_iter=args.max_iter)
        
        self.train_loader = self._create_dataloader(args)
        self.val_loader = self._create_val_dataloader(args)

        self.evaluator = COCO_custom_evaluator(args.data_dir, args.home)


    def _create_model(self, args, model_name):

        self.logger.info("creating model...")

        pixel_mean = args.pixel_mean.split()
        pixel_mean = [float(x) for x in pixel_mean]
        pixel_std = args.pixel_std.split()
        pixel_std = [float(x) for x in pixel_std]

        if model_name == "ResNet-50-FPN":
            model = MaskRCNN(args, pixel_mean, pixel_std)
        
        if args.load:
            model, loaded_layers = load_my_rpn(model, args.load_name)
            for key in loaded_layers:
                self.logger.info("loaded layer {} from pretrained model".format(key))
        
        return model


    def _get_optimizer(self, model, args):
        if args.optimizer == "sgd-default":
            return SGD(model.parameters(), args.lr)

        elif args.optimizer == "sgd-custom":
            # What is used in detectron2/plain_train_net.py
            # To be implemented based on detectron2/solver/build.py/build_optimizer and get_default_optimizer_params
            raise NotImplementedError("sgd-cusgom used get_default_optimizer_params not implemented yet")

        else:
            raise ValueError("unknown optimizer")


    def _get_lr_scheduler(self, lr, lr_scheduler, start_iter, max_iter):
        if lr_scheduler == "multistep":
            return MultiStepLR(self.optimizer, milestones=self.args.lr_steps, gamma=0.1)
            
        elif lr_scheduler == "multistep-warmup":
            # What is used in detectron2/plain_train_net.py
            # To be implemented based on detectron2/solver/lr_scheduler.py/WarmupMultiStepLR
            raise NotImplementedError("multistep-warmup not implemented yet")

        elif lr_scheduler == "cosine":
            raise NotImplementedError("CosineAnnealingWarmRestarts not implemented yet")

        else:
            raise ValueError("unknown scheduler")
    

    def _create_dataloader(self, args):
        return build_dataloader(args, train=True)

    def _create_val_dataloader(self, args):
        return build_dataloader(args, train=False)


    def train(self):
        start_iter = self.args.start_iter
        _iter = start_iter
        max_iter = self.args.max_iter

        mkdir(os.path.join(self.args.home, "vis"))

        self.train_timer.tic()

        current_best_mAP = -1

        self.logger.info("===================================================Training start===================================================")
        # args printing here
        self.logger.info("Using {} training data, {} evaluation data".format(len(self.train_loader.dataset), len(self.val_loader.dataset)))
        self.logger.debug("All settings used:")

        for k in hyperparams:
            self.logger.debug("{}: {}".format(k, vars(self.args)[k]))

        self.logger.info("====================================================================================================================")
        self.logger.info("")

        while _iter < max_iter:
            for i, data in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
                output, loss_dict, extra = self.model(batched_imgs, image_sizes, annotations, image_ids, is_training=True)
                losses = {k : v.sum() for k, v in loss_dict.items()}

                loss = losses["loss_rpn_cls"] + losses["loss_rpn_loc"] * 10 + losses["loss_cls"] + losses["loss_box_reg"]

                pos_score, neg_score = extra
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()


                if (i + _iter) % 100 == 0 and (i + _iter) != 0:
                    total_time, avg_time = self.train_timer.toc()
                    ETA = (avg_time * self.args.max_iter) / 100
                    ETA = ETA - total_time

                    h, m, s = sec2minhrs(ETA)
                    h2, m2, s2 = sec2minhrs(total_time)
                    print("iter: {}, avg_time: {} s/iter, elapsed_time: {} h {} m {} s, ETA: {} h {} m {} s"
                        .format((i + _iter), round(avg_time / 100, 4), h2, m2, s2, h, m, s))
                    
                    self.logger.debug("iter: {}, avg_time: {} s/iter, elapsed_time: {}h {}m {}s, ETA: {}h {}m {}s"
                        .format((i + _iter), round(avg_time / 100, 4), h2, m2, s2, h, m, s))
                    
                    self.logger.debug("loss_rpn_cls: {}, loss_rpn_loc: {}, loss_cls: {}, loss_box_reg: {}, pos_score: {}, neg_score: {}"
                        .format(round(float(losses["loss_rpn_cls"]), 3), round(float(losses["loss_rpn_loc"]), 3), 
                        round(float(losses["loss_cls"]), 3),
                        round(float(losses["loss_box_reg"]), 3),
                        round(float(pos_score.mean()) * 100,2), 
                        round(float(neg_score.mean()) * 100,2)))
                
                if (i + _iter) % 2000 == 0 and (i + _iter) != 0:
                    print("evaluating...")
                    mAP = self.eval((i + _iter))
                    torch.save(self.model.state_dict(), os.path.join(self.args.home, "last.pkl"))
                    if mAP > current_best_mAP:
                        # save model
                        torch.save(self.model.state_dict(), os.path.join(self.args.home, "best.pkl"))
                        # to load, 
                        # model.load_state_dict(torch.load(PATH))
                        self.logger.warning("Current best mAP: {}, saving model into {}".format(round(mAP, 4), os.path.join(self.args.home, "best.pkl")))
                        current_best_mAP = mAP
                    self.logger.warning("================================================================================================================")
                    self.logger.warning("")

                if (i + _iter) == max_iter:
                    break
            
            print("epoch done, returning...")
            _iter += i
        print("Finished Training.")


    def eval(self, iter_):
        self.logger.warning("")
        self.logger.warning("===================================================EVALUATION===================================================")
        self.model.eval()
        self.eval_timer.tic()
        save_iter = iter_ % 367

        result_boxes = []
        gt_boxes = []
        img_ids = []
        _id = 0
        for i, data in enumerate(self.val_loader):
            if i % 20 == 0 and i != 0:
                print(i, " / 674")
            batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
            with torch.no_grad():
                output, _, _ = self.model(batched_imgs, image_sizes, annotations, image_ids, is_training=False)
            ret_boxes, ret_scores, _, num_inferences = output
            num_inferences = num_inferences.tolist()
            ret_boxes = ret_boxes.split(num_inferences)
            ret_scores = ret_scores.split(num_inferences)

            for image_id, boxes_per_image, scores_per_image in zip(image_ids, ret_boxes, ret_scores):
                for box, score in zip(boxes_per_image.tolist(), scores_per_image.tolist()):
                    # box to x, y, width, height form
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
            
            if i == save_iter:
                # visualize
                vis_gt_and_prop(batched_imgs[0], [x["annotations"] for x in data][0], ret_boxes[0], "bbox", "anchor", 
                    self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[0])) + "_" + "_proposal.jpeg")
                vis_gt_and_prop(batched_imgs[1], [x["annotations"] for x in data][1], ret_boxes[1], "bbox", "anchor", 
                    self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[1])) + "_" + "_proposal.jpeg")
                vis_gt_and_prop(batched_imgs[2], [x["annotations"] for x in data][2], ret_boxes[2], "bbox", "anchor", 
                    self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[2])) + "_" + "_proposal.jpeg")
                vis_gt_and_prop(batched_imgs[3], [x["annotations"] for x in data][3], ret_boxes[3], "bbox", "anchor", 
                    self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[3])) + "_" + "_proposal.jpeg")

        
        gt = {"annotations": gt_boxes, "images": img_ids, "categories": [{"supercategory": "person", "id": 1, "name": "person"}]}
        stats, strings = self.evaluator.evaluate(gt, result_boxes)
        mAP, mAR = stats[0], stats[6]

        for string in strings:
            self.logger.warning(string)

        eval_time, _ = self.eval_timer.toc()
        self.logger.warning("iter: {}, eval time: {} s".format(iter_, round(eval_time, 2)))
        return mAP


    def test(self):
        result_boxes = []
        gt_boxes = []
        img_ids = []
        _id = 0

        for i, data in enumerate(self.val_loader):
            if i % 20 == 0 and i != 0:
                print(i, " / 100")
            batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
            # with torch.no_grad():
            #     output, _, _ = self.model(batched_imgs, image_sizes, annotations, image_ids, is_training=False)

            # ret_boxes, ret_scores, _, num_inferences = output
            # num_inferences = num_inferences.tolist()
            # ret_boxes = ret_boxes.split(num_inferences)
            # ret_scores = ret_scores.split(num_inferences)

            # # store predictions
            # for image_id, boxes_per_image, scores_per_image in zip(image_ids, ret_boxes, ret_scores):
            #     for box, score in zip(boxes_per_image.tolist(), scores_per_image.tolist()):
            #         # box to x, y, width, height form
            #         bbox = {
            #             "image_id": int(image_id),
            #             "category_id": 1,
            #             "bbox": box_to_eval_form(box),
            #             "score": score
            #         }
            #         result_boxes.append(bbox)
            # store gt boxes
            for image_id, gt_boxes_per_image in zip(image_ids, [x["annotations"] for x in data]):
                for box in gt_boxes_per_image.tolist():
                    _id = _id + 1
                    # box to x, y, width, height form
                    gt_boxes.append(box_to_gt_form(box, image_id, _id))
                    img_ids.append({"id": int(image_id)})
                    bbox = {
                        "image_id": int(image_id),
                        "category_id": 1,
                        "bbox": box,
                        "score": 1
                    }
                    result_boxes.append(bbox)

            if i == 100:
                break
        
        gt = {"annotations": gt_boxes, "images": img_ids, "categories": [{"supercategory": "person", "id": 1, "name": "person"}]}
        stats, strings = self.evaluator.evaluate(gt, result_boxes)
        for string in strings:
            print(string)



    def match(self, batched_imgs, annotations, output, image_ids, iter_, vis=False):
        l = int(len(annotations)/4)
        gt_boxes = [annotations[:l], annotations[l:2*l], annotations[2*l:3*l], annotations[3*l:]]
        gt_boxes = [remove_zero_gt(x).clone().to(image_ids.device) for x in gt_boxes]

        proposals = [output[0][:1000], output[1][:1000], output[0][1000:], output[1][1000:]]
        proposals = [x.clone().to(image_ids.device) for x in proposals]

        top_matches = [find_top_match_proposals(gt_box, proposal, image_id) for gt_box, proposal, image_id in zip(gt_boxes, proposals, image_ids)]

        top_match_iou = [x[1] for x in top_matches]
        top_matches = [x[0] for x in top_matches]

        cnt = 0
        tot = 0

        for i in top_match_iou:
            for j in i:
                if j > 0.4:
                    cnt += 1
                tot += 1

        if vis:
            vis_gt_and_prop(batched_imgs[0], gt_boxes[0], top_matches[0], "bbox", "anchor", 
                self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[0])) + "_" + "_proposal.jpeg")
            vis_gt_and_prop(batched_imgs[1], gt_boxes[1], top_matches[1], "bbox", "anchor", 
                self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[1])) + "_" + "_proposal.jpeg")
            vis_gt_and_prop(batched_imgs[2], gt_boxes[2], top_matches[2], "bbox", "anchor", 
                self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[2])) + "_" + "_proposal.jpeg")
            vis_gt_and_prop(batched_imgs[3], gt_boxes[3], top_matches[3], "bbox", "anchor", 
                self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[3])) + "_" + "_proposal.jpeg")
        return cnt, tot


def box_to_eval_form(box):
    """
    input box [x1, y1, x2, y2]
    output box [x1, y1, width, height]
    """
    box = [round(x, 2) for x in box]
    return [box[0], box[1], box[2]-box[0], box[3]-box[1]]


def box_to_gt_form(box, image_id, idx):

    # "area": 702.1057499999998,
    # "iscrowd": 0,
    # "image_id": 289343,
    # "bbox": [473.07,395.93,38.65,28.67],
    # "category_id": 18,
    # "id": 1768

    bbox = [round(x, 2) for x in box]
    area = box[2] * box[3]

    return {
        "area": area,
        "image_id": int(image_id),
        "bbox": bbox,
        "category_id": 1,
        "iscrowd": 0,
        "id": idx
    }




def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--is_train', type=bool, default=True)
    
    _parser.add_argument('--num_gpus', type=int, default=2)
    _parser.add_argument('--gpu_ids', type=str, 
                         help="Use comma between ids")
    _parser.add_argument('--home', type=str, default="./output/test09")
    _parser.add_argument('--num_workers', type=int, default=4)

    _parser.add_argument('--model_name', type=str, default="ResNet-50-FPN")

    _parser.add_argument('--optimizer', type=str, default="sgd-default")
    _parser.add_argument('--lr_scheduler', type=str, default="multistep", 
                          help="multistep, multistep-warmup, cosine")
    _parser.add_argument('--image_per_batch', type=int, default=2)
    _parser.add_argument('--batch_size', type=int, default=4)
    

    _parser.add_argument('--min_size', type=str, default='640 672 704 736 768 800')
    _parser.add_argument('--max_size', type=int, default=1333)

    _parser.add_argument('--data_dir', type=str, 
                         default="/media/thanos_hdd0/taeryunglee/detectron2/coco")

    _parser.add_argument('--pixel_mean', type=str, default='103.53 116.28 123.675')
    _parser.add_argument('--pixel_std', type=str, default='1.0 1.0 1.0')
    _parser.add_argument('--fpn_out_chan', type=int, default=256)

    _parser.add_argument('--load_pretrained_resnet', type=bool, default=True)
    _parser.add_argument('--pretrained_resnet', type=str, default="pretrained/R-50.pkl")


    # Hyperparameters
    _parser.add_argument('--load', type=bool, default=True)
    _parser.add_argument('--load_name', type=str, default="./output/test03/best.pkl")
    
    _parser.add_argument('--freeze_backbone', type=bool, default=False)
    _parser.add_argument('--freeze_rpn', type=bool, default=False)

    _parser.add_argument('--start_iter', type=int, default=0)
    _parser.add_argument('--max_iter', type=int, default=360000)
    _parser.add_argument('--lr_steps', type=tuple, default=(240000, 320000))
    _parser.add_argument('--lr', type=float, default=0.005/4)
    
    _parser.add_argument('--rpn_pos_weight', type=float, default=1.0)
    _parser.add_argument('--roi_pos_weight', type=float, default=1.0)

    _parser.add_argument('--rpn_nms_thresh', type=float, default=0.7)
    _parser.add_argument('--rpn_nms_topk_train', type=int, default=2000)
    _parser.add_argument('--rpn_nms_topk_test', type=int, default=1000)
    _parser.add_argument('--rpn_nms_topk_post', type=int, default=1000)

    _parser.add_argument('--rpn_batch_size', type=int, default=256)
    _parser.add_argument('--roi_batch_size', type=int, default=256)
    
    _parser.add_argument('--roi_test_score_thresh', type=float, default=0.5)
    _parser.add_argument('--roi_nms_thresh', type=float, default=0.4)
    _parser.add_argument('--roi_nms_topk_post', type=int, default=25)
    
    args = _parser.parse_args()
    return args


def load_my_rpn(model, filename):
    with open(filename, "rb") as f:
        data = torch.load(f)

    pretrained_state_dict = data
    model_state_dict = model.state_dict()
    
    dict_to_feed = {rename_layer(k): v for k, v in pretrained_state_dict.items()}
    
    model_state_dict.update(dict_to_feed)

    model.load_state_dict(model_state_dict)

    return model, dict_to_feed.keys()


def rename_layer(name):
    if name.startswith("module."):
        return name[7:]


def main():
    print(sys.argv)
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    # trainer.test()

if __name__ == "__main__":
    main()
