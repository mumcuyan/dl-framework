from modules.sequential import Sequential
from modules.layers import Linear, Dropout

from modules.losses import LossMSE, LossCrossEntropy
from optimizers.sgd import SGD

"""
this module contains list of functions for possible model construction examples
"""


def get_network(x_all, y_all,
                num_of_hidden_layers=3,
                loss='ce',
                num_of_neurons=(2, 25, 25, 25, 2),
                activation='relu',
                lr=0.1,
                momentum_coef=0.0,
                weight_decay=0.0,
                p_dropout=0.0,
                num_of_epochs=100,
                val_split=0.2,
                verbose=0):

    """
    creates model with given parameters
    
    x_all - features
    y_all - targets
    num_of_hidden_layers - int, number of hidden layers in model
    loss - 'ce' for Cross Entropy, 'mse' for Mean Squared Error
    num_of_neurons - tuple of ints with size num_of_hidden_layers + 2, 
                            first element is number of features in x_all and last element is number of possible targets
    activation - 'relu' for ReLu, 'tanh' for Tanh
    lr - float, learning rate
    momentum_coef - float in range (0, 1), momentum coefficient
    weight_decay - float, L2-regularization parameter
    p_dropout - float in range [0, 1), probability of dropout
    num_of_epochs - int, number of epochs
    val_split - float in range [0, 1), ratio of validation set
    verbose - 0 or 1, for printing out results
    """
    
    # set loss and last activation
    if loss == 'ce':
        loss = LossCrossEntropy()
        last_activation = 'softmax'
    else:
        loss = LossMSE()
        last_activation = activation
    
    # initialize empty Sequential as model
    model = Sequential()
    
    # add linear layers with given activations and dropout layers after linear modules with given p_dropout
    if num_of_hidden_layers > 0:
        model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
        model.add(Dropout(prob=p_dropout))
    
        for i in range(num_of_hidden_layers-1):
            model.add(Linear(out=num_of_neurons[i+2], activation=activation))
            model.add(Dropout(prob=p_dropout))

        model.add(Linear(out=num_of_neurons[-1], activation=last_activation))
    
    else:
        model.add(Linear(out=num_of_neurons[-1], input_size=num_of_neurons[0], activation=last_activation))
    
    # set loss of model
    model.loss = loss

    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_ce_1(x_all, y_all,
                    num_of_neurons=(2, 25, 2),
                    activation='relu',
                    lr=0.1,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):

    """
    1 hidden layer, CE
    """
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
    model.add(Dropout(prob=p_dropout))

    model.add(Linear(out=num_of_neurons[2], activation='softmax'))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    # initialize SGD optimizer with given learning rate, momentum coefficient and weight decay parameter
    sgd = SGD(lr, momentum_coef, weight_decay)
    
    # train model, take report

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)
    
    # return model and report
    return model, report


def get_network_ce_2(x_all, y_all,
                  num_of_neurons=(2, 25, 2),
                  activation='relu',
                  lr=0.1,
                  momentum_coef=0.0,
                  weight_decay=0.0,
                  p_dropout=0.0,
                  num_of_epochs=100,
                  val_split=0.2,
                  verbose=0):
    """
    model with 1 hidden layer, loss is CE
    """
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
    model.add(Dropout(prob=p_dropout))

    model.add(Linear(out=num_of_neurons[2], activation='softmax'))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_ce_3(x_all, y_all,
                    num_of_neurons=(2, 25, 25, 2),
                    activation='relu',
                    lr=0.1,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):
    """
    model with 2 hidden layers, loss is CE
    """
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[2], activation=activation))
    model.add(Dropout(prob=p_dropout))

    model.add(Linear(out=num_of_neurons[3], activation='softmax'))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_ce_4(x_all, y_all,
                    num_of_neurons=(2, 25, 25, 25, 2),
                    activation='relu',
                    lr=0.1,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):
    """
    model with 3 hidden layers, loss is CE
    """
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[2], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[3], activation=activation))
    model.add(Dropout(prob=p_dropout))
    
    model.add(Linear(out=num_of_neurons[4], activation='softmax'))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_mse_1(x_all, y_all,
                    num_of_neurons=(2, 25, 2),
                    activation='relu',
                    lr=0.001,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):
    """
    model with 1 hidden layer, loss is MSE
    """
    mse = LossMSE()
    
    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[2], activation=activation))

    model.loss = mse
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_mse_2(x_all, y_all,
                    num_of_neurons=(2, 25, 25, 2),
                    activation='relu',
                    lr=0.001,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):
    """
    model with 2 hidden layers, loss is MSE
    """
    mse = LossMSE()
    
    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[2], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[3], activation=activation))

    model.loss = mse
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def get_network_mse_3(x_all, y_all,
                    num_of_neurons=(2, 25, 25, 25, 2),
                    activation='relu', lr=0.001,
                    momentum_coef=0.0,
                    weight_decay=0.0,
                    p_dropout=0.0,
                    num_of_epochs=100,
                    val_split=0.2,
                    verbose=0):
    """
    model with 3 hidden layers, loss is MSE
    """
    mse = LossMSE()
    
    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[2], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[3], activation=activation))
    model.add(Dropout(prob=p_dropout))
    model.add(Linear(out=num_of_neurons[4], activation=activation))

    model.loss = mse
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report
