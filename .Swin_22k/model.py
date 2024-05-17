import torch
import timm
from torch import nn
from config import CONFIG

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
                CONFIG.BACKBONE,
                num_classes=CONFIG.N_TARGETS,
                pretrained=True)

    def forward(self, inputs):
        return self.backbone(inputs)
