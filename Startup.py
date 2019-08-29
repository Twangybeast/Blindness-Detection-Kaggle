import os

import torch
from PIL import Image, ImageFile

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


INPUT_ROOT = r'input'

MODEL_ROOT = r'models'
MODEL_NAME = r'mobilenet_04'
MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)


VALIDATION_PERCENTAGE = 0.1

# TODO figure out if large images work
IMG_SIZE = 512

NUM_CLASSES = 5

BATCH_SIZE = 16
EPOCHS = 100
NUM_WORKERS = 8

LEARNING_RATE = 1e-3
L2_LOSS = 1e-5

