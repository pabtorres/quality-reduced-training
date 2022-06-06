import torch
import torch.nn as nn
import torchvision.models as models
from thesis_neural_networks_imagenet import *

# Optimizadores y Schedulers
import torch.optim as optim

class EfficientNetB3():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = EfficientNetB3_20_classes()
        else: 
            self.net = EfficientNetB3_1000_classes()
        self.lr = 1e-4
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, momentum=0.9, alpha=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.97)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        return f'FullImageNetCheckpoint/effnetb3/effnetb3_1000_checkpoint'

    def get_checkpoint_acc(self):
        return 65.804

    def get_checkpoint_lr(self):
        return 8.079828447811299e-05

def instantiate_network(selection):
    if selection=="effnetb3_1000":
        return EfficientNetB3(out_features=1000)