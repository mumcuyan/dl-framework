import torch
from networks import default_net_1, default_net_2, default_net_3, default_net_4, default_net_5, default_net_6
from generate_data import generate_data
from utils import transform_classification_labels, one_hot_label

"""

def transform_classification_labels(one_hot_labels, val=-1):
  transformed_labels = one_hot_labels.clone()
  transformed_labels[one_hot_labels <= 0] = val
  
  return transformed_labels

def one_hot_label(labels, num_of_classes=None):
  if num_of_classes is None:
    num_of_classes = int(labels.max()) + 1 # assuming class labels are 0, 1, 2, ... n-1
    
  labels_one_hot = FloatTensor(labels.shape[0], num_of_classes).fill_(0)
  labels_one_hot.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1.0)
  
  return labels_one_hot
"""

points, labels = generate_data(is_torch=True, num_of_points=1000)
print(type(points), " -- ", type(labels))
labels = transform_classification_labels(one_hot_label(labels))

seq1, loss1 = default_net_1(points, labels, num_of_neurons=(2,25,25,25,2), lr=0.1, momentum_coef=0.0, num_of_epochs=1000)
seq2, loss2 = default_net_2(points, labels, num_of_neurons=(2,25,2), lr=0.1, momentum_coef=0.0, num_of_epochs=1000)
# seq3, loss3 = default_net_3(points, transform_classification_labels(one_hot_label(labels), val=0), num_of_neurons=(2,25,2), lr=100, momentum_coef=0.0, num_of_epochs=1000)
# seq4, loss4 = default_net_4(points, transform_classification_labels(one_hot_label(labels), val=0), num_of_neurons=(2,2), lr=100, momentum_coef=0.0, num_of_epochs=1000)
# seq5, loss5 = default_net_5(points, transform_classification_labels(one_hot_label(labels)), num_of_neurons=(2,2), lr=0.1, momentum_coef=0.0, num_of_epochs=1000)
# seq6, loss6 = default_net_6(points, transform_classification_labels(one_hot_label(labels), val=-1), num_of_neurons=(2,25,2), lr=0.1, momentum_coef=0.0, num_of_epochs=10000)