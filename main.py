import numpy as np
import torch


def forward(tensor_in: torch.FloatTensor, do_dropout=True):

    # dropout will effect tensor_in
    print("Shape: {}".format(tensor_in.shape))
    vals = np.random.binomial(1, 0.2, size=tensor_in.shape[1])
    print(vals.shape, " -- ", vals)
    k = torch.mv(tensor_in, torch.from_numpy(vals).type(torch.FloatTensor))
    print("Vals: {}".format(k))

    return vals

#forward(torch.FloatTensor([[1, 2], [3, 4]]))

p = 0.5  # probability of keeping a unit active. higher = less dropout

def train_step(X, W):
    """ X contains the data """
    b1 = b2 = b3 = np.array([[0, 0]])
    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W.T, X) + b1)
    print("H1: ", H1)
    U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask
    print("U1: ", U1)
    H1 *= U1  # drop!
    print("H1: ", H1)
    H2 = np.maximum(0, np.dot(W, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p  # second dropout mask
    H2 *= U2  # drop!
    out = np.dot(W, H2) + b3

X = np.array([[1, 2], [2, 3]])
W = np.array([[4], [5]])

train_step(X, W)
