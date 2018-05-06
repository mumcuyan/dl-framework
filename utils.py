import torch

"""
def transform_classification_labels(one_hot_labels: torch.LongTensor, val=-1):
    transformed_labels = one_hot_labels.clone()
    transformed_labels[one_hot_labels <= 0] = val

    return transformed_labels
"""


def label2one_hot(labels, num_of_classes=None, val=-1):

    if num_of_classes is None:
        num_of_classes = int(labels.max()) + 1  # assuming class labels are 0, 1, 2, ... n-1

    labels_one_hot = torch.FloatTensor(labels.shape[0], num_of_classes).fill_(0)
    labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)

    labels_one_hot[labels_one_hot <= 0] = val  # e.g -1, -1, 1, -1
    return labels_one_hot


def one_hot2label(y_vals: torch.FloatTensor):
    return y_vals.max(1)[1]
