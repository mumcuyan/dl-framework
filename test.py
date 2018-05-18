import argparse
from utils import label2one_hot
from utils.generate_data import generate_data
from modules import Linear, Sequential
from modules.losses import LossCrossEntropy, LossMSE
from optimizers import SGD


def get_categorical_model(input_neurons, output_neurons, layers=None):
    """
    creates a model with Categorical Crossentropy Loss
    :param input_neurons: input neuron number
    :param output_neurons: output neuron number
    :param layers: list of intermediate neuron sizes, default is the number of neurons and layer sizes for neuron
    :return: network with Categorical Crossentropy loss
    """
    if layers is None:
        layers = [25, 25, 25]

    default_act = 'relu'
    model = Sequential()

    idx = 1
    layers.insert(0, input_neurons)
    while idx < len(layers):
        model.add(Linear(out=layers[idx], input_size=layers[idx - 1], activation=default_act))
        idx += 1

    # model.add(Dropout(prob=0.2))
    model.add(Linear(out=output_neurons, activation='softmax'))

    # Set loss function to model: Sequential object
    ce = LossCrossEntropy()
    model.loss = ce
    return model


def get_mse_model(input_neurons, output_neurons, layers=None):
    """
    creates a model with MSE loss
    :param input_neurons: input neuron number
    :param output_neurons: output neuron number
    :param layers: list of intermediate neuron sizes, default is the number of neurons and layer sizes for neuron
    :return: network with MSE loss
    """
    if layers is None:
        layers = [25, 25, 25]

    default_act = 'tanh'
    model = Sequential()

    idx = 1
    layers.insert(0, input_neurons)
    while idx < len(layers):
        model.add(Linear(out=layers[idx], input_size=layers[idx - 1], activation=default_act))
        idx += 1

    # model.add(Dropout(prob=0.2))
    model.add(Linear(out=output_neurons, activation='tanh'))

    # Set loss function to model: Sequential object
    mse = LossMSE()
    model.loss = mse
    return model


def train(model, train_dataset, test_dataset):

    (x_train, y_train) = train_dataset
    (x_test, y_test) = test_dataset

    lr = 0.1
    momentum_coef = 0
    weight_decay = 0

    print(model)

    opt = SGD(lr=lr, momentum_coef=momentum_coef, weight_decay=weight_decay)
    print('Optimizer: {} with (lr: {} -- momentum_coef: {} -- weight_decay: {})'.
          format(opt.__class__.__name__, lr, momentum_coef, weight_decay))

    num_of_epochs = 1000
    batch_size = 256
    val_split = 0.1
    print('Validation Split: {} -- BatchSize: {} -- Epochs: {}'.format(val_split, batch_size, num_of_epochs))
    print('Training is about the start with epoch: {}, batch_size: {}, validation_split: {}'
          .format(num_of_epochs, batch_size, val_split))

    opt.train(model,
              x_train, y_train,
              num_of_epochs=num_of_epochs,
              batch_size=batch_size,
              val_split=val_split,
              verbose=1)

    print('\nEvaluating with test dataset !..')

    test_acc, test_loss = model.evaluate(x_test, y_test, return_pred=False)
    train_acc, train_loss = model.evaluate(x_train, y_train, return_pred=False)
    print("train_acc: {} -- test_loss: {}".format(train_acc, train_loss))
    print("test_acc: {} -- test_loss: {}".format(test_acc, test_loss))

    print('For complete use case of the framework please refer to guide.ipynb')


def main():

    parser = argparse.ArgumentParser(description='BCI-Project: 5 models are available.')
    parser.add_argument("-l", action='store', default="mse", type=str,  # default
                        required=False, help="mse(default) or ce")
    args = parser.parse_args()
    model_type = args.l

    if model_type == 'ce':
        model = get_categorical_model(input_neurons=2, output_neurons=2)
        one_hot_val = 0
    elif model_type == 'mse':
        model = get_mse_model(input_neurons=2, output_neurons=2)
        one_hot_val = -1
    else:
        raise ValueError('Given model_type {} is invalid'.format(model_type))

    x_train, y_train_label = generate_data(num_of_points=1000)
    y_train = label2one_hot(y_train_label, val=one_hot_val)  # convert labels to 1-hot encoding

    x_test, y_test_label = generate_data(num_of_points=1000)
    # x_test, y_test_label = generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51)
    y_test = label2one_hot(y_test_label, val=one_hot_val)

    train(model, (x_train, y_train), (x_test, y_test))


main()
