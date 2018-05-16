import math
import torch


def get_labels(points: torch.FloatTensor, center: torch.FloatTensor, radius: float):
    """

    :param points:
    :param center:
    :param radius:
    :return:
    """
    num_of_points = points.shape[0]

    labels = torch.FloatTensor(num_of_points).fill_(1)
    labels[torch.pow(points - center, 2).sum(1) <= radius ** 2] = 0
    return labels


def generate_data(minn=0, maxx=1, radius=1/math.sqrt(2 * math.pi), center=(0.5, 0.5), num_of_points=1000):
    """

    :param minn:
    :param maxx:
    :param radius:
    :param center:
    :param num_of_points:
    :return:
    """

    center = torch.FloatTensor([center[0], center[1]])
    points = torch.Tensor(num_of_points, 2).uniform_(minn, maxx)
    return points, get_labels(points, center, radius)


def __generate_grid(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    h = x.shape[0]
    w = y.shape[0]
    grid = torch.stack([x.repeat(w), y.repeat(h, 1).t().contiguous().view(-1)],1)
    return grid


def generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51, radius=1/math.sqrt(2 * math.pi)):
    """

    :param minn:
    :param maxx:
    :param num_of_points_per_dim:
    :param radius:
    :return:
    """

    coor_x = torch.linspace(minn, maxx, num_of_points_per_dim)
    coor_y = torch.linspace(minn, maxx, num_of_points_per_dim)

    points = __generate_grid(coor_x, coor_y)

    val = (minn + maxx) / 2
    center = torch.FloatTensor([val, val])

    return points, get_labels(points, center, radius)



