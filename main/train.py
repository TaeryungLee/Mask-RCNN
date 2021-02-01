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
from utils.timer import sec2minhrs

def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--is_train', type=bool, default=True)
    
    _parser.add_argument('--num_gpus', type=int, default=2)
    _parser.add_argument('--gpu_ids', type=str, 
                         help="Use comma between ids")
    _parser.add_argument('--home', type=str, default="./output/test03")
    _parser.add_argument('--num_workers', type=int, default=4)

    _parser.add_argument('--model_name', type=str, default="ResNet-50-FPN")
    _parser.add_argument('--max_iter', type=int, default=270000)
    _parser.add_argument('--start_iter', type=int, default=0)
    
    _parser.add_argument('--lr', type=int, default=0.00001)
    _parser.add_argument('--optimizer', type=str, default="sgd-default")
    _parser.add_argument('--lr_scheduler', type=str, default="multistep", 
                          help="multistep, multistep-warmup, cosine")
    _parser.add_argument('--image_per_batch', type=int, default=2)
    _parser.add_argument('--batch_size', type=int, default=4)
    

    _parser.add_argument('--load', type=bool, default=False)
    _parser.add_argument('--load_name', type=str, default="")
    

    _parser.add_argument('--min_size', type=str, default='640 672 704 736 768 800')
    _parser.add_argument('--max_size', type=int, default=1333)

    _parser.add_argument('--data_dir', type=str, 
                         default="/media/thanos_hdd0/taeryunglee/detectron2/coco")

    _parser.add_argument('--pixel_mean', type=str, default='103.53 116.28 123.675')
    _parser.add_argument('--pixel_std', type=str, default='1.0 1.0 1.0')
    _parser.add_argument('--fpn_out_chan', type=int, default=256)

    _parser.add_argument('--load_pretrained_resnet', type=bool, default=True)
    _parser.add_argument('--pretrained_resnet', type=str, default="pretrained/R-50.pkl")
    
    args = _parser.parse_args()
    return args

class Trainer(DefaultTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.args = args
        self.model = self._create_model(args, model_name=args.model_name)

        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.optimizer = self._get_optimizer(model=self.model, args=args)

        self.lr_scheduler = self._get_lr_scheduler(
            lr=args.lr, lr_scheduler=args.lr_scheduler,
            start_iter=args.start_iter, max_iter=args.max_iter)
        
        self.train_loader = self._create_dataloader(args)
        self.val_loader = self._create_val_dataloader(args)


    def _create_model(self, args, model_name):
        pixel_mean = args.pixel_mean.split()
        pixel_mean = [float(x) for x in pixel_mean]
        pixel_std = args.pixel_std.split()
        pixel_std = [float(x) for x in pixel_std]

        if model_name == "ResNet-50-FPN":
            return MaskRCNN(args, pixel_mean, pixel_std)


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
            return MultiStepLR(self.optimizer, milestones=[210000, 250000], gamma=0.1)
            
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

        logger = Logger(self.args.home)


        # freeze_layer_dict = self.model.module.loaded_layers
        # freeze_layer = [freeze_layer_dict[key] for key in freeze_layer_dict.keys()]
        # for name, param in self.model.module.backbone.bottom_up.named_parameters():
        #     if name in freeze_layer:
        #         param.requires_grad = False
        current_best_mAP = -1

        while _iter < max_iter:
            for i, data in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
                output, loss_dict, pos_l, neg_l = self.model(batched_imgs, image_sizes, annotations, image_ids)
                losses = {k : v.sum() for k, v in loss_dict.items()}
                loss = losses["loss_rpn_cls"] + losses["loss_rpn_loc"] * 10
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if (i + _iter) % 100 == 0:
                    total_time, avg_time = self.train_timer.toc()
                    ETA = (avg_time * self.args.max_iter) / 100
                    ETA = ETA - total_time

                    h, m, s = sec2minhrs(ETA)
                    print("iter: {}, avg_time: {}s/iter, ETA: {}h {}m {}s, loss_rpn_cls: {}, loss_rpn_loc: {}"
                        .format((i + _iter), round(avg_time / 100, 4), h, m, s, round(float(losses["loss_rpn_cls"]), 3), round(float(losses["loss_rpn_loc"]), 3)))
                    
                    logger.debug("iter: {}, avg_time: {}s/iter, ETA: {}h {}m {}s"
                        .format((i + _iter), round(avg_time / 100, 4), h, m, s))
                    
                    logger.debug("loss_rpn_cls: {}, loss_rpn_loc: {}, pos_logit: {}, neg_logit: {}"
                        .format(round(float(losses["loss_rpn_cls"]), 3), round(float(losses["loss_rpn_loc"]), 3), 
                        round(float(pos_l.mean()) * 100,2), 
                        round(float(neg_l.mean()) * 100,2)))
                
                if (i + _iter) % 2000 == 0 and (i + _iter) != 0:
                    print("evaluating...")
                    mAP = self.eval_rpn((i + _iter), logger)
                    if mAP > current_best_mAP:
                        # save model
                        torch.save(self.model.state_dict(), os.path.join(self.args.home, "best.pkl"))
                        # to load, 
                        # model.load_state_dict(torch.load(PATH))
                        logger.warning("Current best mAP: {}, saving model into {}".format(round(mAP, 4), os.path.join(self.args.home, "best.pkl")))
                        current_best_mAP = mAP

                if (i + _iter) == max_iter:
                    break
            
            print("returning...")
            _iter += i
        print("Finished Training.")

    
    def eval_rpn(self, iter_, logger):
        self.model.eval()
        eval_stats = {
            "loss_rpn_cls": [],
            "loss_rpn_loc": [],
            "pos_l": [],
            "neg_l": [],
            "tp": [],
            "fp": []
        }
        save_iter = iter_ % 367
        self.eval_timer.tic()

        for i, data in enumerate(self.val_loader):
            if i % 20 == 0 and i != 0:
                print(i)
            batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
            output, loss_dict, pos_l, neg_l = self.model(batched_imgs, image_sizes, annotations, image_ids)
            losses = {k : v.sum() for k, v in loss_dict.items()}
            cnt, tot = self.match(batched_imgs, annotations, output, image_ids, iter_, vis=(i==save_iter))

            eval_stats["loss_rpn_cls"].append(float(losses["loss_rpn_cls"]))
            eval_stats["loss_rpn_loc"].append(float(losses["loss_rpn_loc"]))
            eval_stats["pos_l"].append(float(pos_l.mean()))
            eval_stats["neg_l"].append(float(neg_l.mean()))
            eval_stats["tp"].append(cnt)
            eval_stats["fp"].append(tot-cnt)
        
        loss_cls = sum(eval_stats["loss_rpn_cls"]) / len(eval_stats["loss_rpn_cls"])
        loss_loc = sum(eval_stats["loss_rpn_loc"]) / len(eval_stats["loss_rpn_loc"])
        pos_l = sum(eval_stats["pos_l"]) / len(eval_stats["pos_l"])
        neg_l = sum(eval_stats["neg_l"]) / len(eval_stats["neg_l"])
        mAP = sum(eval_stats["tp"]) / (sum(eval_stats["tp"]) + sum(eval_stats["fp"]))
        eval_time, _ = self.eval_timer.toc()

        logger.warning("===================================================EVALUATION===================================================")
        logger.warning("iter: {}, eval time: {}".format(iter_, round(eval_time, 2)))
        logger.warning("avg_loss_cls: {}, avg_loss_loc: {}, mAP: {}".format(round(loss_cls, 3), round(loss_loc, 3), round(mAP, 4)))
        logger.warning("pos_logit: {}, neg_logit: {}".format(round(pos_l * 100, 2), round(neg_l * 100, 2)))
        logger.warning("================================================================================================================")

        return mAP

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


























# ====================== Not used ======================

    def eval(self, iter_, logger, batched_imgs, image_sizes, annotations, image_ids):
        # data = self.val_loader
        self.model.eval()
        output, loss_dict, pos_l, neg_l = self.model(batched_imgs, image_sizes, annotations, image_ids)
        losses = {k : v.sum() for k, v in loss_dict.items()}

        cnt, tot = self.visualize_proposal(batched_imgs, annotations, output, image_ids, iter_)

        logger.info("<EVAL> iter: {}, loss_rpn_cls: {}, loss_rpn_loc: {}, pos_logit: {}, neg_logit: {}, {} / {} ({}%) correctly proposed"
            .format(iter_,  round(float(losses["loss_rpn_cls"]), 3), round(float(losses["loss_rpn_loc"]), 3), 
            round(float(pos_l.mean()) * 100,2), 
            round(float(neg_l.mean()) * 100,2), cnt, tot, round(cnt * 100 / tot, 2)))


    def visualize_proposal(self, batched_imgs, annotations, output, image_ids, iter_):
        l = int(len(annotations)/4)
        gt_boxes = [annotations[:l], annotations[l:2*l], annotations[2*l:3*l], annotations[3*l:]]
        gt_boxes = [remove_zero_gt(x).clone().to(image_ids.device) for x in gt_boxes]

        proposals = [output[0][:1000], output[1][:1000], output[0][1000:], output[1][1000:]]
        proposals = [x.clone().to(image_ids.device) for x in proposals]

        top_matches = [find_top_match_proposals(gt_box, proposal, image_id)[0] for gt_box, proposal, image_id in zip(gt_boxes, proposals, image_ids)]
        top_match_iou = [find_top_match_proposals(gt_box, proposal, image_id)[1] for gt_box, proposal, image_id in zip(gt_boxes, proposals, image_ids)]

        cnt = 0
        tot = 0
        for i in top_match_iou:
            for j in i:
                if j > 0.4:
                    cnt += 1
                tot += 1

        vis_gt_and_prop(batched_imgs[0], gt_boxes[0], top_matches[0], "bbox", "anchor", 
            self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[0])) + "_" + "_proposal.jpeg")
        vis_gt_and_prop(batched_imgs[1], gt_boxes[1], top_matches[1], "bbox", "anchor", 
            self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[1])) + "_" + "_proposal.jpeg")
        vis_gt_and_prop(batched_imgs[2], gt_boxes[2], top_matches[2], "bbox", "anchor", 
            self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[2])) + "_" + "_proposal.jpeg")
        vis_gt_and_prop(batched_imgs[3], gt_boxes[3], top_matches[3], "bbox", "anchor", 
            self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[3])) + "_" + "_proposal.jpeg")

        # vis_gt_and_prop(batched_imgs[0], annotations[:l], output[0][0:1000], "bbox", "anchor", 
        #     self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[0])) + "_" + "_all.jpeg")
        # vis_gt_and_prop(batched_imgs[1], annotations[l:2*l], output[1][0:1000], "bbox", "anchor", 
        #     self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[1])) + "_" + "_all.jpeg")
        # vis_gt_and_prop(batched_imgs[2], annotations[2*l:3*l], output[0][1000:], "bbox", "anchor", 
        #     self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[2])) + "_" + "_all.jpeg")
        # vis_gt_and_prop(batched_imgs[3], annotations[3*l:], output[1][1000:], "bbox", "anchor", 
        #     self.args.home + "/vis/" + str(iter_) + "_" + str(int(image_ids[3])) + "_" + "_all.jpeg")
        return cnt, tot



    def test(self):
        for i, data in enumerate(self.train_loader):
            if i == 1:
                break
            
            print("iteration ", i)
            
            batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)
            output, loss_dict = self.model(batched_imgs, image_sizes, annotations, image_ids)

            # l = int(len(annotations)/4)
            # vis_gt_and_prop(batched_imgs[0], annotations[:l], output[0][0:30], "bbox", "anchor", 
            #     "vis/" + str(i) + "_" + str(int(image_ids[0])) + "_" + "_bbox.jpeg")
            # vis_gt_and_prop(batched_imgs[1], annotations[l:2*l], output[1][0:30], "bbox", "anchor", 
            #     "vis/" + str(i) + "_" + str(int(image_ids[1])) + "_" + "_bbox.jpeg")
            # vis_gt_and_prop(batched_imgs[2], annotations[2*l:3*l], output[0][1000:1030], "bbox", "anchor", 
            #     "vis/" + str(i) + "_" + str(int(image_ids[2])) + "_" + "_bbox.jpeg")
            # vis_gt_and_prop(batched_imgs[3], annotations[3*l:], output[1][1000:1030], "bbox", "anchor", 
            #     "vis/" + str(i) + "_" + str(int(image_ids[3])) + "_" + "_bbox.jpeg")

            # losses = {k : v.sum() for k, v in loss_dict.items()}
            # loss = losses["loss_rpn_cls"] + losses["loss_rpn_loc"] * 10


            # vis_denorm_tensor_with_bbox(batched_imgs[0], output[0][0:30],
            #     "anchor", "vis/" + str(i) + "_" + str(int(image_ids[0])) + "_" + "_bbox.jpeg")
            # vis_denorm_tensor_with_bbox(batched_imgs[1], output[1][0:30],
            #     "anchor", "vis/" + str(i) + "_" + str(int(image_ids[1])) + "_" + "_bbox.jpeg")
            # vis_denorm_tensor_with_bbox(batched_imgs[2], output[0][1000:1030],
            #     "anchor", "vis/" + str(i) + "_" + str(int(image_ids[2])) + "_" + "_bbox.jpeg")
            # vis_denorm_tensor_with_bbox(batched_imgs[3], output[1][1000:1030],
            #     "anchor", "vis/" + str(i) + "_" + str(int(image_ids[3])) + "_" + "_bbox.jpeg")


def main():
    print(sys.argv)
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

    # trainer.test()

if __name__ == "__main__":
    main()
