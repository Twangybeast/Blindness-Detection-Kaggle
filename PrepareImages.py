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


FOLDER_NAME = 'test15_t3_512'

if __name__ == '__main__':
    train_csv = pd.read_csv(os.path.join(INPUT_ROOT, 'testLabels15.csv'))
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
            img_name = os.path.join(INPUT_ROOT, 'test15', train_csv.iat[i, 0] + '.jpg')
            image = Preprocessing.load_twangy_color(img_name, image_size=512)
            cv2.imwrite(os.path.join(INPUT_ROOT, FOLDER_NAME, train_csv.iat[i, 0] + '.png'), image)

            vis.line(Y=[(timeit.default_timer() - time1)], X=[i], win=heartbeat_plot,
                     update=('append' if i else 'replace'))

