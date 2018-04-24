from modules.linear import Linear
from modules.activations import ReLU, Sigmoid, Tanh
from modules.losses import LossMSE, LossSoftmaxCrossEntropy
from modules.optimizers import SGD
from modules.sequential import Sequential


def default_net_1(input, target, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    relu1 = ReLU()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    relu2 = ReLU()
    lin3 = Linear(num_of_neurons[2], num_of_neurons[3])
    relu3 = ReLU()
    lin4 = Linear(num_of_neurons[3], num_of_neurons[4])
    mse = LossMSE(target)  # TODO: what is target and how it's used in loss

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(relu1, name="ReLU1")
    seq.add_module(lin2, name="Lin2")
    seq.add_module(relu2, name="ReLU2")
    seq.add_module(lin3, name="Lin3")
    seq.add_module(relu3, name="ReLU3")
    seq.add_module(lin4, name="Lin4")
    seq.add_module(mse, name="MSE")

    # integrate loss function to optimizer like in Keras
    sgd = SGD(lr, momentum_coef)
    print(type(input), " -- ", type(seq))

    # TODO verbose
    #
    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging


def default_net_2(input, target, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    relu1 = ReLU()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    mse = LossMSE(target)

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(relu1, name="ReLU1")
    seq.add_module(lin2, name="Lin2")
    seq.add_module(mse, name="MSE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging


def default_net_3(input, target, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    sig1 = Sigmoid()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    sig2 = Sigmoid()
    ce = LossSoftmaxCrossEntropy(target)

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(sig1, name="Sigmoid1")
    seq.add_module(lin2, name="Lin2")
    seq.add_module(sig2, name="Sigmoid2")
    seq.add_module(ce, name="CE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, ce.loss_logging


def default_net_4(input, target, num_of_neurons=(2, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    sig1 = Sigmoid()
    ce = LossSoftmaxCrossEntropy(target)

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(sig1, name="Sigmoid1")
    seq.add_module(ce, name="CE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, ce.loss_logging


def default_net_5(input, target, num_of_neurons=(2, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    mse = LossMSE(target)

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(mse, name="MSE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging


def default_net_6(input, target, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    tan1 = Tanh()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    tan2 = Tanh()
    mse = LossMSE(target)

    seq = Sequential()
    seq.add_module(lin1, name="Lin1")
    seq.add_module(tan1, name="Tan1")
    seq.add_module(lin2, name="Lin2")
    seq.add_module(tan2, name="Tan2")
    seq.add_module(mse, name="MSE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging