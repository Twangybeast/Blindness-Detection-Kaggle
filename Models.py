from torch import nn
from torchvision import models

from Startup import *

def generate_model_mobilenet():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    return model
