import math
import torch
import matplotlib.pyplot as plt


def shuffle(x_data, y_data):
    perm_ = torch.randperm(x_data.shape[0])
    return x_data[perm_], y_data[perm_]


def batch(x_train, y_train, batch_size, is_shuffle=True):
    if is_shuffle:
        x_train, y_train = shuffle(x_train, y_train)

    data_size = x_train.shape[0]
    for start in range(0, data_size, batch_size):
        end = min(start+batch_size, data_size)
        yield x_train[start: end], y_train[start: end]


def split_data(x_all, y_all, val_split, is_shuffle=True):
    if is_shuffle:
        x_all, y_all = shuffle(x_all, y_all)

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


def prepare_standardplot(title, xlabel, figsize=(10,6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)
    ax1.set_ylabel('categorical cross entropy')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2


def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)


def plot_report(train_report, title="", figsize=(10,6), is_save_fig=False, filename=''):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch', figsize)
    ax1.plot(train_report["train_loss"], label = "train")
    ax1.plot(train_report["val_loss"], label = "validation")
    ax2.plot(train_report["train_acc"], label = "train")
    ax2.plot(train_report["val_acc"], label = "validation")
    finalize_standardplot(fig, ax1, ax2)

    if is_save_fig:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')

    return fig
