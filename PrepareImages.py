import os

import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import Preprocessing
from Startup import *


# Super janky method to utilize Dataset's parallelization in order to preprocess all the images

class PrepareImages(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = Preprocessing.transform_ndarray2tensor()
        self.col_id = self.data.columns.get_loc('id_code')      # should be 0
        self.col_label = self.data.columns.get_loc('diagnosis')     # should be 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img_name = os.path.join(INPUT_ROOT, 'train_images', self.data.iat[index, self.col_id] + '.png')
        image = Preprocessing.load_ben_color(img_name)
        cv2.imwrite(os.path.join(INPUT_ROOT, 'train_images_cropped', self.data.iat[index, self.col_id] + '.png'), image)
        return 0


if __name__ == '__main__':
    train_csv = pd.read_csv(os.path.join(INPUT_ROOT, 'train.csv'))
    ds = PrepareImages(train_csv)
    dl = DataLoader(ds, batch_size=16, num_workers=8)
    with tqdm(range(len(dl))) as pbar:
        for x in dl:
            pbar.update()
