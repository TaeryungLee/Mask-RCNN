import logging
import os
import sys
import abc
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.optim import SGD, Adam
import argparse
from utils.default import DefaultTrainer


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--num_gpus', type=int, default=2)
    _parser.add_argument('--gpu_ids', type=str, required=True, 
                          help="Use space between ids")
    _parser.add_argument('--home', type=str, default="./output/test01")
    _parser.add_argument('--num_workers', type=int, default=8)

    _parser.add_argument('--model_name', type=str, default="ResNet-50-FPN")
    _parser.add_argument('--max_iter', type=int, default=80000)
    _parser.add_argument('--start_iter', type=int, default=0)
    _parser.add_argument('--load', type=bool, default=False)
    
    _parser.add_argument('--optimizer', type=str, default="sgd-default")

    _parser.add_argument('--lr', type=int, default=0.005)
    _parser.add_argument('--lr_scheduler', type=str, default="multistep", 
                          help="multistep, multistep-warmup, cosine")
    
    _parser.add_argument('--image_per_batch', type=int, default=2)
    _parser.add_argument('--batch_size', type=int, default=1024)
    
    args = _parser.parse_args()
    return args

class Trainer(DefaultTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.model = self._create_model(model_name=args.model_name)

        self.optimizer = self.get_optimizer(model=self.model, args=args)

        self.lr_scheduler = self._get_lr_scheduler(
            lr=args.lr, lr_scheduler=args.lr_scheduler,
            start_iter=args.start_iter, max_iter=args.max_iter)
        
        self.train_loader = self._create_dataloader()
                

    def _create_model(self, model_name):
        # Based on detectron2/modeling/meta_arch/rcnn.py/GeneralizedRCNN
        pass


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
    

    def _create_dataloader(self):
        pass


    def train(self):
        pass










def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
