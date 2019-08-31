import os
import timeit

import cv2
import pandas as pd
import visdom

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import Preprocessing
from Startup import *


# Super janky method to utilize Dataset's parallelization in order to preprocess all the images

FOLDER_NAME = 'train_images_t2_512'

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
        image = Preprocessing.load_twangy_color(img_name)
        cv2.imwrite(os.path.join(INPUT_ROOT, FOLDER_NAME, self.data.iat[index, self.col_id] + '.png'), image)
        return 0


if __name__ == '__main__':
    train_csv = pd.read_csv(os.path.join(INPUT_ROOT, 'train.csv'))
    if not os.path.exists(os.path.join(INPUT_ROOT, FOLDER_NAME)):
        os.makedirs(os.path.join(INPUT_ROOT, FOLDER_NAME))
    # run this part first
    #ds = PrepareImages(train_csv)
    #dl = DataLoader(ds, batch_size=16, num_workers=8)
    #with tqdm(range(len(dl))) as pbar:
    #   for x in dl:
    #       pbar.update()
    # then run this part
    vis = visdom.Visdom()
    heartbeat_plot = vis.line(Y=[0], X=[0])
    for i in tqdm(range(len(train_csv))):
        time1 = timeit.default_timer()
        id = train_csv.iat[i, 0]
        if not os.path.isfile(os.path.join(INPUT_ROOT, FOLDER_NAME, id + '.png')):
            img_name = os.path.join(INPUT_ROOT, 'train_images', train_csv.iat[i, 0] + '.png')
            image = Preprocessing.load_twangy_color(img_name)
            cv2.imwrite(os.path.join(INPUT_ROOT, FOLDER_NAME, train_csv.iat[i, 0] + '.png'), image)

            vis.line(Y=[(timeit.default_timer() - time1)], X=[i], win=heartbeat_plot,
                     update=('append' if i else 'replace'))

