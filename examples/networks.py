from modules.sequential import Sequential
from modules.layers import Linear, Dropout

from modules.losses import LossMSE, LossCrossEntropy
from optimizers.sgd import SGD


def default_net(x_all, y_all, num_of_hidden_layers=3, loss='ce', num_of_neurons=(2, 25, 25, 25, 2), activation='relu', lr=0.1, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    generic
    """
    if loss == 'ce':
        loss = LossCrossEntropy()
        last_activation = 'softmax'
    else:
        loss = LossMSE()
        last_activation = activation
        
    model = Sequential()
    
    if num_of_hidden_layers > 0:
        model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation=activation))
        model.add(Dropout(prob=p_dropout))
    
        for i in range(num_of_hidden_layers-1):
            model.add(Linear(out=num_of_neurons[i+2], activation=activation))
            model.add(Dropout(prob=p_dropout))

        model.add(Linear(out=num_of_neurons[-1], activation=last_activation))
    
    else:
        model.add(Linear(out=num_of_neurons[-1], input_size=num_of_neurons[0], activation=last_activation))
    
    model.loss = loss
    sgd = SGD(lr, momentum_coef, weight_decay=weight_decay)

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def default_net_1(x_all, y_all, num_of_neurons=(2, 25, 2), activation='relu', lr=0.1, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
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

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def default_net_1(x_all, y_all, num_of_neurons=(2, 25, 2), activation='relu', lr=0.1, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
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

    report = sgd.train(model, x_all, y_all, num_of_epochs, val_split=val_split, verbose=verbose)

    return model, report


def default_net_2(x_all, y_all, num_of_neurons=(2, 25, 25, 2), activation='relu', lr=0.1, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    2 hidden layers, CE
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


def default_net_3(x_all, y_all, num_of_neurons=(2, 25, 25, 25, 2), activation='relu', lr=0.1, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    3 hidden layers, CE
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


def default_net_4(x_all, y_all, num_of_neurons=(2, 25, 2), activation='relu', lr=0.001, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    1 hidden layer, MSE
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


def default_net_5(x_all, y_all, num_of_neurons=(2, 25, 25, 2), activation='relu', lr=0.001, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    2 hidden layers, MSE
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


def default_net_6(x_all, y_all, num_of_neurons=(2, 25, 25, 25, 2), activation='relu', lr=0.001, momentum_coef=0.0, weight_decay=0.0, p_dropout=0.0, num_of_epochs=100, val_split=0.2, verbose=0):
    """
    3 hidden layers, MSE
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