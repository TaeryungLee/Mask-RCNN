import os
from abc import abstractmethod, ABCMeta
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from utils.dir import mkdir
from utils.timer import Timer
from utils.logger import colorLogger

class DefaultTrainer(metaclass=ABCMeta):
    def __init__(self, args):
        # Initialize training

        # Home directory of training
        self.home = args.home
        mkdir(self.home)

        # Start epoch initially 0, overrided if load
        self.start_iter = 0
        self.current_iter = 0

        # Timer and logger
        # Timer reference: https://github.com/mks0601/I2L-MeshNet_RELEASE/
        self.train_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # self.logger = colorLogger(os.path.join(self.home, "logs"))

    @abstractmethod
    def _create_model(self, model_name):
        pass
    
    @abstractmethod
    def _create_dataloader(self):
        pass

    @abstractmethod
    def _get_optimizer(self, model_name):
        pass

    def _get_lr_scheduler(self, lr, lr_scheduler, start_iter, max_iter):
        pass

    def save_model(self, epoch):
        pass

    def load_model(self, path, model_name):
        pass
    
    def _data_sampler(self):
        pass

    @abstractmethod
    def train(self):
        pass

class DefaultTester():
    def __init__(self):
        pass

