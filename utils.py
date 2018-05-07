import torch


def split_data(x_all, y_all, val_split):
    val_size = int(x_all.shape[0] * val_split)
    train_size = x_all.shape[0] - val_size
    x_train, x_val = torch.split(x_all, [train_size, val_size], dim=0)
    y_train, y_val = torch.split(y_all, [train_size, val_size], dim=0)

    return (x_train, y_train), (x_val, y_val)


def label2one_hot(labels, num_of_classes=None, val=0):

    if num_of_classes is None:
        num_of_classes = int(labels.max()) + 1  # assuming class labels are 0, 1, 2, ... n-1

    labels_one_hot = torch.FloatTensor(labels.shape[0], num_of_classes).fill_(0)
    labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)

    labels_one_hot[labels_one_hot <= 0] = val  # e.g -1, -1, 1, -1
    return labels_one_hot


def one_hot2label(y_vals: torch.FloatTensor):
    return y_vals.max(1)[1]
