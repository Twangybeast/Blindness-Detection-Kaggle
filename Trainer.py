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


class BlindnessTrainDataset(Dataset):
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
        image = self.transform(image)
        label = torch.tensor(self.data.iat[index, self.col_label])
        return image, label

def get_data_properties(dataframe):
    counts = dataframe['diagnosis'].value_counts(normalize=True)
    freqs = [0] * 5
    for i in range(NUM_CLASSES):
        freqs[i] = counts[i]
    return {'class_freqs': freqs}

def load_training_datasets():
    train_csv = pd.read_csv(os.path.join(INPUT_ROOT, 'train.csv'))
    # Shuffle and split to train & evaluation
    train_csv = train_csv.sample(frac=1, random_state=139847)
    split_boundary = int(len(train_csv) * EVALUATION_PERCENTAGE)
    train_df, eval_df = train_csv[split_boundary:], train_csv[:split_boundary]

    train_ds, eval_ds = BlindnessTrainDataset(train_df), BlindnessTrainDataset(eval_df)
    return train_ds, eval_ds, get_data_properties(train_csv)

def loss_func(data_properties):
    # TODO try continuous kappa loss function
    weights = 1/torch.tensor(data_properties['class_freqs'])
    weights = weights.to(device)
    return nn.CrossEntropyLoss(weights)

def fit(model, optimizer, scheduler, criterion, train_dl, eval_dl, epochs=EPOCHS):
    # TODO print kappa
    for epoch in range(epochs):
        print(f'Epoch: {epoch}/{epochs}')
        print('-' * 10)
        model.train()

        running_loss = 0.0
        counter = 0
        with tqdm(range(len(train_dl))) as tk0:
            for inputs, labels in train_dl:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                counter += inputs.size(0)
                tk0.update()
                tk0.set_postfix(loss=(running_loss / counter))

        epoch_loss = running_loss / len(train_dl)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        model.eval()
        with torch.no_grad():
            predictions_list = []
            labels_list = []
            running_loss = 0.0
            counter = 0
            for inputs, labels in eval_dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                loss = criterion(output, labels)

                running_loss += loss.item() * inputs.size(0)
                counter += inputs.size(0)

                predictions = torch.argmax(output, dim=-1)
                predictions_list.append(predictions)
                labels_list.append(labels)
            all_predictions = torch.cat(predictions_list)
            all_labels = torch.cat(labels_list)
            kappa_score = Utils.compute_kappa(all_predictions.cpu().numpy(), all_labels.cpu().numpy())
            print('Evaluation Kappa: {:.4f}'.format(kappa_score))

            eval_loss = running_loss / len(eval_dl)
            print('Evaluation Loss: {:.4f}'.format(eval_loss))

        scheduler.step()
        print(f'Current LR: {scheduler.get_lr()}')
        torch.save(model, MODEL_PATH)

def main():
    train_ds, eval_ds, data_properties = load_training_datasets()
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    model = Models.generate_model_mobilenet()
    # model = torch.load(MODEL_PATH)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LOSS)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    with Timer('Finished training in {}') as _:
        fit(model, optimizer, scheduler, loss_func(data_properties), train_dl, eval_dl)


if __name__ == '__main__':
    main()
