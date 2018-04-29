import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_data(minn=0, maxx=1, radius=1 / np.sqrt(2 * np.pi), center=np.array([0.5, 0.5]), num_of_points=1000,
                  is_torch=False):
    points = np.random.uniform(low=minn, high=maxx, size=(num_of_points, 2))
    labels = (np.sum(np.square(center - points), axis=1) <= np.square(radius)).astype(np.float)

    if is_torch:
        points = torch.from_numpy(points)
        labels = torch.from_numpy(labels)
        return points.type(torch.FloatTensor), labels.type(torch.FloatTensor)

    return points, labels


def generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51, is_torch=False):
    coor_x = np.linspace(minn, maxx, num_of_points_per_dim)
    coor_y = np.linspace(minn, maxx, num_of_points_per_dim)
    xx, yy = np.meshgrid(coor_x, coor_y)
    points = np.array([xx.flatten(), yy.flatten()]).T

    if is_torch:
        points = torch.from_numpy(points)

        return points.type(torch.FloatTensor)

    return points


def plot_data(points, labels=0, minn=0, maxx=1, radius=1 / np.sqrt(2 * np.pi), center=np.array([0.5, 0.5])):
    circle = plt.Circle(center, radius, color='r', fill=False, linewidth=5)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    ax.set_xlim(minn, maxx)
    ax.set_ylim(minn, maxx)


def main():
    generate_data(is_torch=True)
