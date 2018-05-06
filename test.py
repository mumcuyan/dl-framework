from utils import label2one_hot
from generate_data import generate_data, generate_grid_data
from modules import Dropout
from modules import Linear, Sequential
from modules.losses import LossCrossEntropy, LossMSE
from optimizers import SGD


def ce_net_1(x_all, y_all, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], input_size=num_of_neurons[1], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], activation='relu'))
    model.add(Dropout(prob=0.2))
    model.add(Linear(out=num_of_neurons[4], activation='softmax'))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=0.2)

    res = sgd.train(model, x_all, y_all, num_of_epochs, val_split=0.2)

    return res, model


def mse_net_1(x_all, y_all, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):

    mse = LossMSE()
    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Linear(out=num_of_neurons[2] + 2, activation='relu'))
    model.add(Linear(out=num_of_neurons[3], activation='relu'))
    model.add(Linear(out=num_of_neurons[4]))
    model.loss = mse

    sgd = SGD(lr, momentum_coef, weight_decay=0)
    sgd.train(model, x_all, y_all, num_of_epochs, val_split=0.2, verbose=1)

    return model


def cat_entropy():
    x_all, y_all = generate_data(num_of_points=500)
    y_all = label2one_hot(y_all, val=0)  # convert labels to 1-hot encoding

    train_report, model = ce_net_1(x_all, y_all, num_of_epochs=2000)
    # loss1 = model.loss.loss_logging
    for key, val in train_report.items():
        print("key: {} -- size: {}".format(key, len(val)))

    x_test, y_test = generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51)

    results = model.evaluate(x_test, label2one_hot(y_test, val=0), return_pred=True)
    print("results: {}".format(results))


def mse():
    x_all, y_all = generate_data(num_of_points=500)
    y_all = label2one_hot(y_all, val=-1)  # convert labels to 1-hot encoding

    model = mse_net_1(x_all, y_all, num_of_epochs=1000)
    x_test, y_test = generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51)

    print("model: {}".format(model.print_model()))
    results = model.evaluate(x_test, label2one_hot(y_test, val=-1), return_pred=True)
    print("results: {}".format(results))


cat_entropy()
# mse()
