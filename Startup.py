import os

import torch
from PIL import Image, ImageFile

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


#INPUT_ROOT = r'input'
INPUT_ROOT = r'E:\Temp\Datasets\APTOS 2019\input'
#IMAGE_FOLDER = r'train_images_t2_512'
IMAGE_FOLDER = r'test15_t2_512'
TRAINING_CSV = 'testLabels15.csv'
IMAGE_ID = 'image' # id_code
IMAGE_LABEL = 'level' # diagnosis

MODEL_ROOT = r'models'
MODEL_NAME = r'efficientnet_05.pth'
MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)
STATE_PATH = os.path.join(MODEL_ROOT, 'training')

NEW_MODEL = False


VALIDATION_PERCENTAGE = 0.1

# TODO figure out if large images work
IMG_SIZE = 224
#IMG_SIZE = 512

NUM_CLASSES = 5

BATCH_SIZE = 8
STEP_FREQ = 2
EPOCHS = 100
NUM_WORKERS = 8

LEARNING_RATE = 1e-4

L2_LOSS = 0
