import torch
from torchvision import datasets


def convert_to_one_hot_labels(input, target, val=0):
    """
    Given input and target, it returns one-hot encoding form
    """
    
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    # for some activation functions, e.g. relu
    if val == 0:
        ret = (tmp + 1) / 2
    # for some activation functions, e.g. tanh
    if val == -1:
        ret = tmp
    return ret


def load_data(one_hot_labels=False, normalize=False, flatten=True, data_dir=None, cifar=False, full=True, tiny=False,
              val=0):
    """
    This function is provided by instructor to get cifar or mnist dataset using torchvision.dataset package
    """

    if data_dir is None:
        data_dir = './data'

    if cifar is not None and cifar:
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train=True, download=True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train=False, download=True)

        train_input = torch.from_numpy(cifar_train_set.train_data)
        # Dirty hack to handle the change between torchvision 1.0.6 and 1.0.8
        if train_input.size(3) == 3:
            train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        else:
            train_input = train_input.float()
        train_target = torch.LongTensor(cifar_train_set.train_labels)

        test_input = torch.from_numpy(cifar_test_set.test_data).float()
        # Dirty hack to handle the change between torchvision 1.0.6 and 1.0.8
        if test_input.size(3) == 3:
            test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        else:
            test_input = test_input.float()
        test_target = torch.LongTensor(cifar_test_set.test_labels)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)

        train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.train_labels
        test_input = mnist_test_set.test_data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.test_labels

    if flatten:
        train_input = train_input.clone().view(train_input.size(0), -1)
        test_input = test_input.clone().view(test_input.size(0), -1)

    if full:
        if tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, 1000)
            train_target = train_target.narrow(0, 0, 1000)
            test_input = test_input.narrow(0, 0, 1000)
            test_target = test_target.narrow(0, 0, 1000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target, val=val)
        test_target = convert_to_one_hot_labels(test_input, test_target, val=val)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target
