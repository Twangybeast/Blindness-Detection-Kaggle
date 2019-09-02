import time

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
import visdom
from fastai.basic_train import Learner, DataBunch
from fastai.metrics import KappaScore
from fastai.train import lr_find
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
        self.col_id = self.data.columns.get_loc(IMAGE_ID)      # should be 0
        self.col_label = self.data.columns.get_loc(IMAGE_LABEL)     # should be 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img_name = os.path.join(INPUT_ROOT, IMAGE_FOLDER, self.data.iat[index, self.col_id] + '.png')
        image = Preprocessing.load_preprocessed_image(img_name)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(self.data.iat[index, self.col_label])
        return image, label

def get_data_properties(dataframe):
    counts = dataframe[IMAGE_LABEL].value_counts(normalize=True)
    freqs = [0] * 5
    for i in range(NUM_CLASSES):
        freqs[i] = counts[i]
    return {'class_freqs': freqs}

def load_training_datasets():
    train_csv = pd.read_csv(os.path.join(INPUT_ROOT, TRAINING_CSV))
    # Shuffle and split to train & validation
    train_csv = train_csv.sample(frac=1, random_state=139847)
    split_boundary = int(len(train_csv) * VALIDATION_PERCENTAGE)
    train_df, eval_df = train_csv[split_boundary:], train_csv[:split_boundary]

    train_ds, eval_ds = BlindnessTrainDataset(train_df), BlindnessTrainDataset(eval_df)
    return train_ds, eval_ds, get_data_properties(train_csv)

def loss_func(data_properties):
    weights = 1/torch.tensor(data_properties['class_freqs'])
    weights = weights.to(device)
    return nn.CrossEntropyLoss(weights)

def find_lr(model, criterion, train_dl, eval_dl, initial_lr=1e-4, gamma=1.05):
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma)

    vis = visdom.Visdom()
    loss_plot = vis.line(Y=[0], X=[0.])

    running_loss = 0.
    plot_freq = 50
    for epoch in range(5):
        with tqdm(range(len(train_dl))) as pbar:
            for batch, (inputs, labels) in enumerate(train_dl):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()

                running_loss += loss.item()

                optimizer.step()

                pbar.update()
                pbar.set_postfix(lr='%.6f' % scheduler.get_lr()[0], test=model._conv_stem.weight.grad[0][0][0][0].item())
                optimizer.zero_grad()

                if (batch + 1) % plot_freq == 0:
                    vis.line(Y=[running_loss / plot_freq], X=[scheduler.get_lr()[0]], win=loss_plot,
                             update=('append' if batch > plot_freq else 'replace'))
                    scheduler.step()
                    running_loss = 0.

def fit(model, optimizer, scheduler, criterion, train_dl, eval_dl, loss_history, epochs=EPOCHS):
    vis = visdom.Visdom()
    if loss_history is None:
        loss_history = [[0, 0]]
    last_epoch = scheduler.last_epoch
    loss_plot = vis.line(Y=loss_history, X=list(zip(range(last_epoch), range(last_epoch))) if last_epoch else [[0, 0]],
                         opts={'title': f'Training {MODEL_NAME}'})
    for epoch in range(last_epoch, epochs):
        print(f'Epoch: {epoch}/{epochs}')
        print('-' * 10)
        model.train()

        running_loss = 0.0
        counter = 0
        next_step = STEP_FREQ
        with tqdm(range(len(train_dl))) as tk0:
            for inputs, labels in train_dl:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()

                next_step = next_step - 1
                if next_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    next_step = STEP_FREQ

                running_loss += loss.item() * inputs.size(0)
                counter += inputs.size(0)
                tk0.update()
                tk0.set_postfix(loss=('{:.4f}'.format(running_loss / counter)))

        epoch_loss = running_loss / counter
        print('Training Loss: {:.4f}'.format(epoch_loss))

        model.eval()
        with torch.no_grad():
            predictions_list = []
            labels_list = []
            running_loss = 0.0
            counter = 0
            with tqdm(range(len(eval_dl))) as tk1:
                for inputs, labels in eval_dl:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output = model(inputs)
                    loss = criterion(output, labels)

                    running_loss += loss.item() * inputs.size(0)
                    counter += inputs.size(0)

                    predictions = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
                    predictions_list.append(predictions.cpu())
                    labels_list.append(labels.cpu())

                    tk1.update()
            all_predictions = torch.cat(predictions_list)
            all_labels = torch.cat(labels_list)

            predictions = all_predictions.cpu().numpy()
            labels = all_labels.cpu().numpy()
            kappa_score = Utils.compute_kappa(predictions, labels)
            accuracy_score = Utils.compute_accuracy(predictions, labels)
            print('Validation Kappa: {:.4f} Accuracy: {:.4f}'.format(kappa_score, accuracy_score))

            eval_loss = running_loss / counter
            print('Validation Loss: {:.4f}'.format(eval_loss))

        vis.line(Y=[[epoch_loss, eval_loss]], X=[[epoch, epoch]], win=loss_plot,
                 update=('append' if epoch else 'replace'))
        loss_history.append([epoch_loss, eval_loss])
        scheduler.step()
        print(f'Current LR: {scheduler.get_lr()}')
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': torch.tensor(loss_history)
        }, STATE_PATH)

# TODO try using fastai learner
def main():
    train_ds, eval_ds, data_properties = load_training_datasets()
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    if NEW_MODEL:
        model = Models.generate_model_efficientnet()
    else:
        model = Models.load_model_efficientnet(MODEL_PATH)
    model.to(device)

    # learn = Learner(DataBunch(train_dl, eval_dl),
    #                 model,
    #                 loss_func=Utils.KappaLoss(),
    #                 metrics=[KappaScore(weights='quadratic')],
    #                 path='.')
    # learn.fit(5)

    # learn.recorder.plot_losses()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LOSS)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    loss_history = None
    if not NEW_MODEL:
        ckpt = torch.load(STATE_PATH)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        loss_history = ckpt['history'].numpy().tolist()

    with Timer('Finished training in {}') as _:
        fit(model, optimizer, scheduler, Utils.KappaLoss(data_properties['class_freqs']),
            train_dl, eval_dl, loss_history)
        # find_lr(model, Utils.KappaLoss(), train_dl, eval_dl)


if __name__ == '__main__':
    main()
