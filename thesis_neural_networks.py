import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class Resnet152_20_classes(nn.Module):
    def __init__(self):
        super(Resnet152_20_classes,self).__init__()
        self.net = resnet152 = models.resnet152()
        self.classifier_layer = nn.Linear(1000, 20)

    def forward(self,x):
        y = self.net(x)
        y = self.classifier_layer(y)
        return {'hidden': y, 'logits': y}

class EfficientNetB3_1000_classes(nn.Module):
    def __init__(self):
        super(EfficientNetB3_1000_classes,self).__init__()
        self.net = effnetb3 = models.efficientnet_b3()

    def forward(self,x):
        y = self.net(x)
        return {'hidden': y, 'logits': y}

class EfficientNetB3_20_classes(nn.Module):
    def __init__(self):
        super(EfficientNetB3_20_classes,self).__init__()
        self.net = effnetb3 = models.efficientnet_b3()
        self.classifier_layer = nn.Linear(1000, 20)

    def forward(self,x):
        y = self.net(x)
        y = self.classifier_layer(y)
        return {'hidden': y, 'logits': y}

class SqueezeNet_20_classes(nn.Module):
    def __init__(self):
        super(SqueezeNet_20_classes,self).__init__()
        self.net = squeezenet = models.squeezenet1_0()
        self.classifier_layer = nn.Linear(1000, 20)

    def forward(self,x):
        y = self.net(x)
        y = self.classifier_layer(y)
        return {'hidden': y, 'logits': y}