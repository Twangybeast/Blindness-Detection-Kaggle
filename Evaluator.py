import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
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
        #self.transform = Preprocessing.transform_ndarray2tensor()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, resample=Image.BICUBIC),  # Arbitrary degree value
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            # normalize the images to torchvision models specifications
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.col_id = self.data.columns.get_loc('id_code')      # should be 0
        self.col_label = self.data.columns.get_loc('diagnosis')     # should be 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img_name = os.path.join(INPUT_ROOT, 'train_images_t3_512', self.data.iat[index, self.col_id] + '.png')
        image = Preprocessing.load_preprocessed_image(img_name)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(self.data.iat[index, self.col_label])
        return image, label


def evaluate(model, dl, data_length):
    model.eval()
    with torch.no_grad():
        aug_count = 10

        predictions = np.zeros(data_length, dtype=np.float)
        for i in range(aug_count):
            predictions_list = []
            labels_list = []
            for inputs, labels in dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs).squeeze(-1)

                #predictions = Utils.predict_class(output)
                prediction = output
                predictions_list.append(prediction)
                labels_list.append(labels)
            all_predictions = torch.cat(predictions_list)
            all_labels = torch.cat(labels_list)

            predictions = predictions + all_predictions.cpu().numpy()
            labels = all_labels.cpu().numpy()

        predictions = predictions / aug_count
        predictions = np.around(np.clip(predictions, 0, NUM_CLASSES - 1)).astype(int)


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
        evaluate(model, eval_dl, len(eval_ds))


if __name__ == '__main__':
    main()