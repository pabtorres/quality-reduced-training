import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB3_1000_classes(nn.Module):
    def __init__(self):
        super(EfficientNetB3_1000_classes,self).__init__()
        self.net = effnetb3 = models.efficientnet_b3(pretrained=True)

    def forward(self,x):
        y = self.net(x)
        return {'hidden': y, 'logits': y}