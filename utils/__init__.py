import torch


def shuffle(x_data, y_data):
    """

    :param x_data:
    :param y_data:
    :return:
    """
    perm_ = torch.randperm(x_data.shape[0])
    return x_data[perm_], y_data[perm_]


def batch(x_train, y_train, batch_size, is_shuffle=True):
    """

    :param x_train:
    :param y_train:
    :param batch_size:
    :param is_shuffle:
    :return:
    """
    if is_shuffle:
        x_train, y_train = shuffle(x_train, y_train)

    data_size = x_train.shape[0]
    for start in range(0, data_size, batch_size):
        end = min(start+batch_size, data_size)
        yield x_train[start: end], y_train[start: end]


def split_data(x_all, y_all, val_split, is_shuffle=True):
    """

    :param x_all:
    :param y_all:
    :param val_split:
    :param is_shuffle:
    :return:
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

    :param labels:
    :param num_of_classes:
    :param val:
    :return:
    """

    if num_of_classes is None:
        num_of_classes = int(labels.max()) + 1  # assuming class labels are 0, 1, 2, ... n-1

    labels_one_hot = torch.FloatTensor(labels.shape[0], num_of_classes).fill_(0)
    labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)

    labels_one_hot[labels_one_hot <= 0] = val  # e.g -1, -1, 1, -1
    return labels_one_hot


def one_hot2label(y_vals: torch.FloatTensor):
    """
    :param y_vals:
    :return:
    """
    return y_vals.max(1)[1]

