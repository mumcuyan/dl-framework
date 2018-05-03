import numpy as np
import torch
import matplotlib.pyplot as plt


def get_labels(points: torch.FloatTensor, center: torch.FloatTensor, radius: float):
    num_of_points = points.shape[0]

    labels = torch.FloatTensor(num_of_points).fill_(1)
    labels[torch.pow(points - center, 2).sum(1) <= radius ** 2] = 0
    return labels


def generate_data(minn=0, maxx=1, radius=1 / np.sqrt(2 * np.pi), center=(0.5, 0.5), num_of_points=1000):

    center = torch.FloatTensor([center[0], center[1]])
    points = torch.Tensor(num_of_points, 2).uniform_(minn, maxx)
    return points, get_labels(points, center, radius)


def generate_grid(x, y):
    h = x.shape[0]
    w = y.shape[0]
    grid = torch.stack([x.repeat(w), y.repeat(h, 1).t().contiguous().view(-1)],1)
    return grid


def generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51, radius=1 / np.sqrt(2 * np.pi)):

    coor_x = torch.linspace(minn, maxx, num_of_points_per_dim)
    coor_y = torch.linspace(minn, maxx, num_of_points_per_dim)

    points = generate_grid(coor_x, coor_y)

    val = (minn + maxx) / 2
    center = torch.FloatTensor([val, val])
    print("ccenter: {}".format(center))
    return points, get_labels(points, center, radius)


def plot_data(points, labels=0, minn=0, maxx=1, radius=1 / np.sqrt(2 * np.pi), center=np.array([0.5, 0.5])):
    circle = plt.Circle(center, radius, color='r', fill=False, linewidth=5)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    ax.set_xlim(minn, maxx)
    ax.set_ylim(minn, maxx)

# generate_data()
generate_grid_data()