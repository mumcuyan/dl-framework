import numpy as np
import matplotlib.pyplot as plt


def plot_data(points, labels=0, minn=0, maxx=1, radius=1 / np.sqrt(2 * np.pi), center=np.array([0.5, 0.5])):
    circle = plt.Circle(center, radius, color='r', fill=False, linewidth=5)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    ax.set_xlim(minn, maxx)
    ax.set_ylim(minn, maxx)


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