from utils import one_hot_label
from generate_data import generate_data

from modules import Linear, Sequential
from modules.activations import ReLU, Sigmoid, Softmax, Tanh
from losses import LossMSE, LossSoftmaxCrossEntropy
from optimizers import SGD


def default_net_1(x_train, y_train, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    model = Sequential(
        [
            Linear(num_of_neurons[0], num_of_neurons[1], activation=ReLU()),
            Linear(num_of_neurons[1], num_of_neurons[2], activation=ReLU()),
            Linear(num_of_neurons[2], num_of_neurons[3], activation=ReLU()),
            Linear(num_of_neurons[3], num_of_neurons[4])
        ]
    )
    # mse = LossMSE()
    ce = LossSoftmaxCrossEntropy()
    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=0.2)

    # TODO verbose
    sgd.train(model, x_train, y_train, num_of_epochs)

    return model, ce.loss_logging


def default_net_2(x_train, y_train, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    relu1 = ReLU()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    mse = LossMSE()

    model = Sequential()
    model.add_module(lin1, name="Lin1")
    model.add_module(relu1, name="ReLU1")
    model.add_module(lin2, name="Lin2")
    model.loss = mse

    sgd = SGD(lr, momentum_coef)
    sgd.train(model, x_train, y_train, num_of_epochs)

    return model, mse.loss_logging


def default_net_3(input, target, num_of_neurons=(2, 25, 2), lr=0.01, momentum_coef=0.0, num_of_epochs=100):
    ce = LossSoftmaxCrossEntropy()
    model = Sequential(
        [
            Linear(num_of_neurons[0], num_of_neurons[1]),
            ReLU(),
            Linear(num_of_neurons[1], num_of_neurons[2]),
        ], loss_func=ce
    )

    sgd = SGD(lr, momentum_coef)

    sgd.train(model, input, target, num_of_epochs)

    return model, ce.loss_logging


points, labels = generate_data(is_torch=True, num_of_points=1000)
print(type(points), " -- ", type(labels))
labels = one_hot_label(labels, val=0)  # convert labels to 1-hot encoding

model, loss1 = default_net_1(points, labels, num_of_epochs=50000)
print(loss1)


"""
    Linear(num_of_neurons[0], num_of_neurons[1]),
    ReLU(),
    Linear(num_of_neurons[1], num_of_neurons[2]),
    ReLU(),
    Linear(num_of_neurons[2], num_of_neurons[3]),
    ReLU(),
    Linear(num_of_neurons[3], num_of_neurons[4])
"""