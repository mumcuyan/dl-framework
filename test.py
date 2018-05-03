from utils import labels2one_hot
from generate_data import generate_data, generate_grid_data
from modules import Dropout
from modules import Linear, Sequential
from modules.losses import LossCrossEntropy
from optimizers import SGD


def default_net_1(x_all, y_all, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    ce = LossCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], activation='relu'))
    model.add(Dropout(prob=0.2))
    model.add(Linear(out=num_of_neurons[4]))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=0.2)

    sgd.train(model, x_all, y_all, num_of_epochs, val_split=0.2)

    return model


x_all, y_all = generate_data(num_of_points=500)
y_all = labels2one_hot(y_all, val=0)  # convert labels to 1-hot encoding


model,  = default_net_1(x_all, y_all, num_of_epochs=1000)
loss1 = model.loss.loss_logging

x_test, y_test = generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51)


out1 = model.predict(x_test)
model.evaluate(x_test, y_test)
