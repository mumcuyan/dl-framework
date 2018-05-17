import math
import torch


def get_labels(points: torch.FloatTensor, center: torch.FloatTensor, radius: float):
    """
    :param points: is a torch.FloatTensor
    :param center: x_coordinate, y_coordinate as tensor of size 2
    :param radius: size of radius as float
    :return: it labels the points inside the circle as 1, otherwise 0
    """
    num_of_points = points.shape[0]
    
    # initially labels are 0
    labels = torch.FloatTensor(num_of_points).fill_(0)
    # set label to 1 if inside circle
    labels[torch.pow(points - center, 2).sum(1) <= radius ** 2] = 1
    return labels


def generate_data(minn=0, maxx=1, radius=1/math.sqrt(2 * math.pi), center=(0.5, 0.5), num_of_points=1000):
    """
    :param minn: minimum value of the range to generate data
    :param maxx: maximum value of the range to generate data
    :param radius: scalar
    :param center: x_coordinate, y_coordinate as tuple
    :param num_of_points: number of points to generate
    :return: torch.FloatTensor of [num_of_points x 2], [num_of_points] as labels
    """
    
    # center as FloatTensor
    center = torch.FloatTensor([center[0], center[1]])
    # generate unifrom points in range [minn, maxx]^2
    points = torch.Tensor(num_of_points, 2).uniform_(minn, maxx)
    # return points and their labels
    return points, get_labels(points, center, radius)


def __generate_grid(x, y):
    """
    :param x: x coordinate values from minn to maxx
    :param y: y coordinate values from minn to maxx
    :return: cross product of each point given for x and y
    """
    h = x.shape[0]
    w = y.shape[0]
    # calculate a grid for given x and y coordinates
    grid = torch.stack([x.repeat(w), y.repeat(h, 1).t().contiguous().view(-1)],1)
    return grid


def generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51, radius=1/math.sqrt(2 * math.pi)):
    """
    :param minn: minimum value of the range to generate data
    :param maxx: maximum value of the range to generate data
    :param num_of_points_per_dim: number of points per dimension (x and y coordinates) to generate
    :param radius: scalar, default value in the projection description
    :return:
    """
    
    # linspace for x and y axises
    coor_x = torch.linspace(minn, maxx, num_of_points_per_dim)
    coor_y = torch.linspace(minn, maxx, num_of_points_per_dim)
    
    # generate gird points for given x and y axises
    points = __generate_grid(coor_x, coor_y)
    
    # calculate the center
    val = (minn + maxx) / 2
    center = torch.FloatTensor([val, val])
    
    # return grid points
    return points, get_labels(points, center, radius)

