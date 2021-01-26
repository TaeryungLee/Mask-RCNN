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


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--num_gpus', type=int, default=2)
    _parser.add_argument('--gpu_ids', type=str, 
                         help="Use comma between ids")
    _parser.add_argument('--home', type=str, default="./output/test01")
    _parser.add_argument('--num_workers', type=int, default=4)

    _parser.add_argument('--model_name', type=str, default="ResNet-50-FPN")
    _parser.add_argument('--max_iter', type=int, default=80000)
    _parser.add_argument('--start_iter', type=int, default=0)
    _parser.add_argument('--load', type=bool, default=False)
    
    _parser.add_argument('--optimizer', type=str, default="sgd-default")

    _parser.add_argument('--lr', type=int, default=0.005)
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
    
    args = _parser.parse_args()
    return args

class Trainer(DefaultTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.args = args
        self.model = self._create_model(args, model_name=args.model_name)

        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        

        # if args.num_gpus > 1:
        #     self.model = nn.DataParallel(self.model)
        
        # self.model.cuda()

        # self.optimizer = self._get_optimizer(model=self.model, args=args)

        # self.lr_scheduler = self._get_lr_scheduler(
        #     lr=args.lr, lr_scheduler=args.lr_scheduler,
        #     start_iter=args.start_iter, max_iter=args.max_iter)
        
        self.train_loader = self._create_dataloader(args)
                

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
            return MultiStepLR(self.optimizer, milestones=[60000, 80000], gamma=0.1)
            
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


    def train(self):
        pass


    def test(self):
        for i, data in enumerate(self.train_loader):
            if i > 20:
                break
            
            print("iteration ", i)

            # print(data[0]["image"])
            # # print("this is last pixel", data[0]["image"][703][1054])
            # print(data[0]["image_id"])

            # vis.vis_numpy(data[0]["image"], "vis/" + str(data[0]["image_id"]) + "_pre" + ".jpeg")
            

            # print(data[2]["image"])
            # # print("this is last pixel", data[2]["image"][735][919])
            # print(data[2]["image_id"])
            # vis.vis_numpy(data[2]["image"], "vis/" + str(data[2]["image_id"]) + "_pre" + ".jpeg")
            
            



            # backbone_features = self.model.module.backbone(batched_imgs)
            # batched_imgs, image_sizes = self.model.preprocess(self.args, data)
            # backbone_features = self.model.backbone(batched_imgs)

            # outputs = self.model(data)

            batched_imgs, image_sizes, annotations, image_ids = self.model.module.preprocess(self.args, data)

            # print(batched_imgs.shape, image_sizes.shape, annotations.shape, image_ids)

            outputs = self.model(batched_imgs, image_sizes, annotations, image_ids)

            # if i % 5 == 0:
            #     torch.cuda.empty_cache()

            




            


def main():
    print(sys.argv)
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    main()
