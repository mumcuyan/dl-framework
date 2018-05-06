from modules.sequential import Sequential
from modules.layers import Linear, Dropout
from modules.activations import ReLU, Tanh
from modules.losses import LossMSE, LossCrossEntropy
from optimizers.sgd import SGD


def default_net_1(input, target, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
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

    mse = LossMSE(target)

    # integrate loss function to optimizer like in Keras
    sgd = SGD(mse, lr, momentum_coef)
    print(type(input), " -- ", type(model))

    sgd.train(model, input, num_of_epochs, verbose=0)

    return model, mse.loss_logging


def default_net_2(input, target, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    relu1 = ReLU()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    mse = LossMSE(target)

    seq = Sequential()
    seq.add(lin1, name="Lin1")
    seq.add(relu1, name="ReLU1")
    seq.add(lin2, name="Lin2")
    seq.add(mse, name="MSE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging


def default_net_3(input, target, num_of_neurons=(2, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    sig1 = Sigmoid()
    lin2 = Linear(num_of_neurons[1], num_of_neurons[2])
    sig2 = Sigmoid()
    ce = LossCrossEntropy(target)

    seq = Sequential()
    seq.add(lin1, name="Lin1")
    seq.add(sig1, name="Sigmoid1")
    seq.add(lin2, name="Lin2")
    seq.add(sig2, name="Sigmoid2")
    seq.add(ce, name="CE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, ce.loss_logging


def default_net_4(input, target, num_of_neurons=(2, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    sig1 = Sigmoid()
    ce = LossCrossEntropy(target)

    seq = Sequential()
    seq.add(lin1, name="Lin1")
    seq.add(sig1, name="Sigmoid1")
    seq.add(ce, name="CE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, ce.loss_logging


def default_net_5(input, target, num_of_neurons=(2, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    lin1 = Linear(num_of_neurons[0], num_of_neurons[1])
    mse = LossMSE(target)

    seq = Sequential()
    seq.add(lin1, name="Lin1")
    seq.add(mse, name="MSE")

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
    seq.add(lin1, name="Lin1")
    seq.add(tan1, name="Tan1")
    seq.add(lin2, name="Lin2")
    seq.add(tan2, name="Tan2")
    seq.add(mse, name="MSE")

    sgd = SGD(lr, momentum_coef)

    sgd.train(input, seq, num_of_epochs)

    return seq, mse.loss_logging