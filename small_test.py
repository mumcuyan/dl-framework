from utils import transform_classification_labels, one_hot_label
from generate_data import generate_data

from modules import Linear, Sequential
from modules.activations import ReLU
from losses import LossMSE
from optimizers import SGD


def default_net_1(x_train, y_train, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    model = Sequential(
        [
            Linear(num_of_neurons[0], num_of_neurons[1]),
            ReLU(),
            Linear(num_of_neurons[1], num_of_neurons[2]),
            ReLU(),
            Linear(num_of_neurons[2], num_of_neurons[3]),
            ReLU(),
            Linear(num_of_neurons[3], num_of_neurons[4])
        ]
    )

    mse = LossMSE()
    model.loss = mse
    # integrate loss function to optimizer like in Keras
    sgd = SGD(lr, momentum_coef)
    print(type(input), " -- ", type(model))

    # TODO verbose
    sgd.train(model, x_train, y_train, num_of_epochs)

    return model, mse.loss_logging


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

points, labels = generate_data(is_torch=True, num_of_points=1000)
print(type(points), " -- ", type(labels))
labels = transform_classification_labels(one_hot_label(labels))

model, loss1 = default_net_2(points, labels, num_of_epochs=1000)
print(loss1)