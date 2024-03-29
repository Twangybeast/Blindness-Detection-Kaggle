import math
import torch
import sklearn.metrics

from Startup import *

# predictions = torch.argmax(output, dim=-1)
def compute_kappa(predictions, labels):
    return sklearn.metrics.cohen_kappa_score(predictions, labels, weights='quadratic')

def compute_accuracy(predictions, labels):
    return sklearn.metrics.accuracy_score(predictions, labels)

class KappaLoss():
    def __init__(self, class_freqs):
        self.confusion_rows = [torch.tensor(0)] * NUM_CLASSES
        self.weights = torch.zeros((NUM_CLASSES, NUM_CLASSES))
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                self.weights[i][j] = float(((i-j)**2)/16)
        self.weights = self.weights.to(device)
        self.softmax = torch.nn.Softmax(dim=-1)

    # Based on this guide
    # https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
    def __call__(self, output, labels):
        output = self.softmax(output)
        # create confusion matrix
        for label in range(NUM_CLASSES):
            self.confusion_rows[label] = \
                torch.index_select(output, 0, torch.nonzero(labels == label).reshape(-1)).sum(dim=0)
        # Confusion matrix O
        O = torch.stack(self.confusion_rows)

        output_hist = torch.sum(O, dim=0)
        labels_hist = torch.sum(O, dim=1)
        E = torch.ger(output_hist, labels_hist)

        O = O / O.sum()
        E = E / E.sum()

        num = (self.weights * O).sum()
        den = (self.weights * E).sum()

        # normally kappa is (1 - (num/den)) so we need to minimize the negative kappa
        kappa = ((num/den) - 1)
        return kappa

def predict_class(output):
    return torch.round(torch.clamp(output, 0, NUM_CLASSES - 1)).to(torch.int64)
