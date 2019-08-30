import os

import torch
from PIL import Image, ImageFile

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


INPUT_ROOT = r'input'

MODEL_ROOT = r'models'
MODEL_NAME = r'efficientnet_01.pth'
MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)
STATE_PATH = os.path.join(MODEL_ROOT, 'training')

NEW_MODEL = True


VALIDATION_PERCENTAGE = 0.1

# TODO figure out if large images work
IMG_SIZE = 256

NUM_CLASSES = 5

BATCH_SIZE = 8
STEP_FREQ = 2
EPOCHS = 100
NUM_WORKERS = 8

LEARNING_RATE = 1e-3
L2_LOSS = 1e-5

