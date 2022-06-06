import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from thesis_neural_networks import *

# Optimizadores y Schedulers
import torch.optim as optim

class EfficientNetB3():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = EfficientNetB3_20_classes()
        else: 
            self.net = models.efficientnet_b3()
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
        return f'F_Checkpoints/effnetb3/effnetb3_checkpoint'

    def get_checkpoint_acc(self):
        return 68.0

    def get_checkpoint_lr(self):
        return 8.329720049289999e-05

class Resnet152():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = Resnet152_20_classes()
        else: 
            self.net = models.resnet152()
        self.lr = 0.01
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        return f'F_Checkpoints/resnet152/resnet152_checkpoint'

    def get_checkpoint_acc(self):
        return 61.6

    def get_checkpoint_lr(self):
        return 0.001

class Squeezenet():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = SqueezeNet_20_classes()
        else: 
            self.net = models.squeezenet1_0()
        self.lr = 0.001
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0002) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        return f'F_Checkpoints/squeezenet/squeezenet_checkpoint'

    def get_checkpoint_acc(self):
        return 39.800000000000004

    def get_checkpoint_lr(self):
        return 1e-05


def instantiate_network(selection):
    if selection=="effnetb3":
        return EfficientNetB3()
    if selection=="squeezenet":
        return Squeezenet()
    if selection=="resnet152":
        return Resnet152()