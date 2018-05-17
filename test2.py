from utils import label2one_hot
from utils.generate_data import generate_data, generate_grid_data
from modules import Dropout
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

    model.add(Dropout(prob=0.2))
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

    model.add(Dropout(prob=0.2))
    model.add(Linear(out=output_neurons, activation='tanh'))

    # Set loss function to model: Sequential object
    mse = LossMSE()
    model.loss = mse
    return model


def train(model, train_dataset, val_dataset):

    (x_train, y_train) = train_dataset

    lr = 0.2
    momentum_coef = 0
    weight_decay = 0.2

    print(model)

    opt = SGD(lr=lr, momentum_coef=momentum_coef, weight_decay=weight_decay)
    print('Optimizer: {} with (lr: {} -- momentum_coef: {} -- weight_decay: {})'.
          format(opt.__class__.__name__, lr, momentum_coef, weight_decay))

    num_of_epochs = 1000
    batch_size = 128
    val_split = 0.1
    print('Training is about the start with epoch: {}, batch_size: {}, validation_split: {}'
          .format(num_of_epochs, batch_size, val_split))

    report = opt.train(model,
              x_train, y_train,
              num_of_epochs=num_of_epochs,
              batch_size=batch_size,
              val_set=val_dataset,
              verbose=1)


def main(model_type='mse'):

    x_train, y_train_label = generate_data(num_of_points=2000)
    y_train = label2one_hot(y_train_label, val=-1)  # convert labels to 1-hot encoding

    x_val, y_val_label = generate_data(num_of_points=1000)
    y_val = label2one_hot(y_val_label, val=-1)  # convert labels to 1-hot encoding

    if model_type == 'ce':
        model = get_categorical_model(input_neurons=x_train.shape[1], output_neurons=y_train.shape[1])
    elif model_type == 'mse':
        model = get_mse_model(input_neurons=x_train.shape[1], output_neurons=y_train.shape[1])
    else:
        raise ValueError('Given model_type {} is invalid'.format(model_type))

    train(model, (x_train, y_train), (x_val, y_val))


main()
