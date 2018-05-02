from utils import one_hot_label
from generate_data import generate_data, generate_grid_data
from modules import Dropout
from modules import Linear, Sequential
from modules.losses import LossSoftmaxCrossEntropy
from optimizers import SGD


def default_net_1(x_train, y_train, num_of_neurons=(2, 25, 25, 25, 2), lr=0.1, momentum_coef=0.0, num_of_epochs=100):
    ce = LossSoftmaxCrossEntropy()

    model = Sequential()
    model.add(Linear(out=num_of_neurons[1], input_size=num_of_neurons[0], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], activation='relu'))
    model.add(Linear(out=num_of_neurons[2], activation='relu'))
    model.add(Dropout(prob=0.2))
    model.add(Linear(out=num_of_neurons[4]))

    model.loss = ce
    sgd = SGD(lr, momentum_coef, weight_decay=0.2)

    sgd.train(model, x_train, y_train, num_of_epochs)

    return model, ce.loss_logging


x_train, y_train = generate_data(is_torch=True, num_of_points=500)
y_train = one_hot_label(y_train, val=0)  # convert labels to 1-hot encoding

print("X_train.type: {} y_train.shape: {}".format(x_train.shape, y_train.shape))
print("y_train: {}".format(y_train))
model, loss1 = default_net_1(x_train, y_train, num_of_epochs=1000)


x_test, y_test = generate_grid_data(minn=0, maxx=1, num_of_points_per_dim=51, is_torch=True)


out1 = model.predict(x_test)
model.evaluate(x_test, y_test)
