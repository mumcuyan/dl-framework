from typing import *
from matplotlib import gridspec
import torch
import math
import matplotlib.pyplot as plt


def plot_data_multiple(list_of_points: List,
                       list_of_labels: List,
                       grid_space: Tuple[int, int],
                       filename: str = None,
                       titles=None,
                       messages=None):
    """
    This function plots multiple datasets including both (points and labels)
    :param list_of_points: list of datapoints to plot
    :param list_of_labels: list of labels to plot
    :param grid_space: is a tuple to how to place figures in the .ipynb
    :param filename: optional paramter if given it is written to disk
    :param titles: list of titles
    :param messages: list of descriptions
    assert len(messages) == len(titles) == len(list_of_points) == len(list_of_labels)

    :return: plot of figures
    """

    assert len(list_of_points) == len(list_of_labels)

    x_num, y_num = grid_space[0], grid_space[1]
    plt.figure(figsize=(8 * y_num, 8 * x_num))
    gs = gridspec.GridSpec(x_num, y_num)

    for idx, (points, labels) in enumerate(zip(list_of_points, list_of_labels)):
        title = '' if titles is None else titles[idx]
        msg = '' if messages is None else messages[idx]

        ax = plt.subplot(gs[idx])  # create subplot based on grid space
        draw_points_circle(ax, points, labels, title=title, msg=msg)

    if filename is not None:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def draw_points_circle(ax, points, labels, title, msg='', minn=0, maxx=1):
    """
    this function plots circle as well as scatters points with labels (with colors)
    :param ax: given subplot
    :param points: points N x 2 (x and y coor)
    :param labels: list of 0s and 1s
    :param title: titles of subplot
    :param msg: description of subplot
    :param minn:
    :param maxx:
    :return: subplot
    """
    radius = 1 / math.sqrt(2 * math.pi)
    center = (0.5, 0.5)

    circle = plt.Circle(center, radius, color='r', fill=False, linewidth=5)
    ax.add_patch(circle)

    if torch.is_tensor(points):
        points = points.numpy()
        ax.scatter(points[:, 0], points[:, 1], c=labels)
    # ax.set_xlim(minn, maxx)
    # ax.set_ylim(minn, maxx)

    ax.set_title(title, fontsize=24)
    ax.set_xlabel(msg)

    for tick_x, tick_y in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
        tick_x.label.set_fontsize(18)
        tick_y.label.set_fontsize(18)


def plot_data(points, labels, title=None):
    """
    same as multiple_plot_data, but one subplot drawn
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_points_circle(ax, points, labels, title)


def plot_report(train_report, title="", figsize=(10, 6), is_save_fig=False, filename=''):

    fig, ax1, ax2 = prepare_standardplot(title, 'epoch', figsize)
    ax1.plot(train_report["train_loss"], label="train")
    ax1.plot(train_report["val_loss"], label="validation")
    ax2.plot(train_report["train_acc"], label="train")
    ax2.plot(train_report["val_acc"], label="validation")
    finalize_standardplot(fig, ax1, ax2)

    if is_save_fig:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')

    return fig


def prepare_standardplot(title, xlabel, figsize=(10, 6)):
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


def plot_v2(train_report, title=""):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(train_report["train_loss"], label="train")
    ax1.plot(train_report["val_loss"], label="validation")
    ax2.plot(train_report["train_acc"], label="train")
    ax2.plot(train_report["val_acc"], label="validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig
