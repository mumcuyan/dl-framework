import torch

def transform_classification_labels(one_hot_labels: torch.LongTensor, val=-1):
    transformed_labels = one_hot_labels.clone()
    transformed_labels[one_hot_labels <= 0] = val

    return transformed_labels


def one_hot_label(labels, num_of_classes=None):

    if num_of_classes is None:
        num_of_classes = int(labels.max()) + 1  # assuming class labels are 0, 1, 2, ... n-1

    labels_one_hot = torch.FloatTensor(labels.shape[0], num_of_classes).fill_(0)
    labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)

    return labels_one_hot