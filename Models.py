from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


from Startup import *

def generate_model_mobilenet():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
        nn.Softmax(dim=-1)
    )
    return model

def generate_model_efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
    return model

def load_model_efficientnet(path):
    model = generate_model_efficientnet()
    model.load_state_dict(torch.load(path))
    return model
