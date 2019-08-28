import torch
import sklearn.metrics

from Startup import *

# predictions = torch.argmax(output, dim=-1)
def compute_kappa(predictions, labels):
    return sklearn.metrics.cohen_kappa_score(predictions, labels, weights='quadratic')

def compute_accuracy(predictions, labels):
    return (predictions == labels).mean()

class KappaLoss():
    def __init__(self):
        self.confusion_rows = [torch.tensor(0)] * NUM_CLASSES
        self.weights = torch.zeros((NUM_CLASSES, NUM_CLASSES))
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                self.weights[i][j] = float(((i-j)**2)/16)
        self.O = torch.zeros((NUM_CLASSES, NUM_CLASSES))
        self.E = torch.zeros((NUM_CLASSES, NUM_CLASSES))

    # Based on this guide
    # https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
    def __call__(self, output, labels):
        # create confusion matrix
        for label in range(NUM_CLASSES):
            self.confusion_rows[label] = \
                torch.index_select(output, 0, torch.nonzero(labels == label).reshape(-1)).sum(dim=0)
        # Confusion matrix O
        torch.stack(self.confusion_rows, out=self.O)

        output_hist = torch.sum(self.O, dim=0)
        labels_hist = torch.sum(self.O, dim=1)
        torch.ger(output_hist, labels_hist, out=self.E)

        self.O = self.O / self.O.sum()
        self.E = self.E / self.E.sum()

        num = (self.weights * self.O).sum()
        den = (self.weights * self.E).sum()

        kappa = (1 - (num/den))
        return kappa
