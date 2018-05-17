import torch


def shuffle(x_data, y_data):
    """
    This function shuffles x_data and y_data with the same permutation
    :param x_data: torch.FloatTensor
    :param y_data: torch.FloatTensor
    """
    perm_ = torch.randperm(x_data.shape[0])
    return x_data[perm_], y_data[perm_]


def batch(x_train, y_train, batch_size, is_shuffle=True):
    """
    This function is a generator for training procedure, given batch size
    it only loads the data to the memory, using python generator to reduce memory footprint
    :param x_train: 2D (N x feature_size) torch.FloatTensor
    :param y_train: 2D (N x class_number) torch.FloatTensor
    :param batch_size: batch-size is an integer
    :param is_shuffle: boolean flag whether to shuffle before splitting
    :return: x_train and y_train, size of batch_size
    """
    if is_shuffle:
        x_train, y_train = shuffle(x_train, y_train)

    data_size = x_train.shape[0]
    for start in range(0, data_size, batch_size):
        end = min(start+batch_size, data_size)
        yield x_train[start: end], y_train[start: end]


def split_data(x_all: torch.FloatTensor, y_all: torch.FloatTensor, val_split: float, is_shuffle=True):
    """
    :param x_all: torch.FloatTensor
    :param y_all: torch.FloatTensor
    :param val_split: is a ratio between 0 and 1
    :param is_shuffle: boolean flag
    :return: (train_dataset), (test_dataset) split by x_all and y_all
    """

    if is_shuffle:
        x_all, y_all = shuffle(x_all, y_all)

    val_size = int(x_all.shape[0] * val_split)
    train_size = x_all.shape[0] - val_size
    x_train, x_val = torch.split(x_all, [train_size, val_size], dim=0)
    y_train, y_val = torch.split(y_all, [train_size, val_size], dim=0)

    return (x_train, y_train), (x_val, y_val)


def label2one_hot(labels, num_of_classes=None, val=0):
    """
    :param labels: list of values
    :param num_of_classes: number of unique classe
    :param val: remaining value other 1, For example 0001000 or -1-1-11-1-1-1
    :return: one-hot encoded version of labels (torch.FloatTensor size of N x number_of_classes)
    """

    if num_of_classes is None:
        num_of_classes = int(labels.max()) + 1  # assuming class labels are 0, 1, 2, ... n-1

    labels_one_hot = torch.FloatTensor(labels.shape[0], num_of_classes).fill_(0)
    labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)

    labels_one_hot[labels_one_hot <= 0] = val  # e.g -1, -1, 1, -1
    return labels_one_hot


def one_hot2label(y_vals: torch.FloatTensor):
    """
    :param y_vals: one-hot form of data
    :return: returns the indices of max value for each row as true label
    """
    return y_vals.max(1)[1]  # works because 1 is always the maximum value given for a row of data

