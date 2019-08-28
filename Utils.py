import torch
import sklearn.metrics

# predictions = torch.argmax(output, dim=-1)
def compute_kappa(predictions, labels):
    return sklearn.metrics.cohen_kappa_score(predictions, labels, weights='quadratic')

def compute_accuracy(predictions, labels):
    return (predictions == labels).mean()
