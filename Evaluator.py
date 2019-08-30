import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

import Models
import Preprocessing
import Utils

from Startup import *
from Timer import Timer
import Trainer

class BlindnessMagicDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = Preprocessing.transform_ndarray2tensor()
        self.col_id = self.data.columns.get_loc('id_code')      # should be 0
        self.col_label = self.data.columns.get_loc('diagnosis')     # should be 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img_name = os.path.join(INPUT_ROOT, 'train_images_t1_512', self.data.iat[index, self.col_id] + '.png')
        image = Preprocessing.load_preprocessed_image(img_name)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(self.data.iat[index, self.col_label])
        return image, label


def evaluate(model, dl):
    model.eval()
    with torch.no_grad():
        predictions_list = []
        labels_list = []
        for inputs, labels in dl:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)

            predictions = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
            predictions_list.append(predictions)
            labels_list.append(labels)
        all_predictions = torch.cat(predictions_list)
        all_labels = torch.cat(labels_list)

        predictions = all_predictions.cpu().numpy()
        labels = all_labels.cpu().numpy()
        kappa_score = Utils.compute_kappa(predictions, labels)
        accuracy_score = Utils.compute_accuracy(predictions, labels)
        print('Evaluation Kappa: {:.4f} Accuracy: {:.4f}'.format(kappa_score, accuracy_score))
        return all_predictions

def main():
    train_ds, eval_ds, data_properties = Trainer.load_training_datasets()
    eval_ds = BlindnessMagicDataset(eval_ds.data)
    eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = Models.load_model_efficientnet(MODEL_PATH)
    model.to(device)

    print('Beginning evaluation')
    with Timer('Finished evaluation in {}') as _:
        evaluate(model, eval_dl)


if __name__ == '__main__':
    main()