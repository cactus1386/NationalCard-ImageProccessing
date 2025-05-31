import torch
import torch.nn as nn
import torchvision.models as models

class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 8) 

    def forward(self, x):
        return self.backbone(x)
