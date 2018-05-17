from unittest import  TestCase
from utils import label2one_hot
from utils.generate_data import generate_data
from examples.networks import default_net_1


class NetworkTest(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """  categorical dataset """
        self.x_train_cat, y_train_label = generate_data(num_of_points=1000)
        self.y_train_cat = label2one_hot(y_train_label, val=0)  # convert labels to 1-hot encoding

        self.x_val_cat, y_train_label = generate_data(num_of_points=750)
        self.y_val_cat = label2one_hot(y_train_label, val=0)  # convert

        self.x_test_cat, y_test_label = generate_data(num_of_points=500)
        self.y_test_cat = label2one_hot(y_test_label, val=0)  # convert

        """ mse dataset """
        self.x_train_mse, y_train_label = generate_data(num_of_points=1000)
        self.y_train_mse = label2one_hot(y_train_label, val=-1)  # convert

        self.x_val_mse, y_train_label = generate_data(num_of_points=750)
        self.y_val_mse = label2one_hot(y_train_label, val=-1)  # convert

        self.x_test_mse, y_test_label = generate_data(num_of_points=500)
        self.y_test_mse = label2one_hot(y_test_label, val=-1)  # convert


    def test_example_1(self):
        pass

    def test_example_2(self):
        pass

    def test_example_3(self):
        pass

    def test_example_4(self):
        pass

    def test_example_5(self):
        pass

    def test_example_6(self):
        pass

    def test_example_7(self):
        pass

    def test_example_8(self):
        pass


    def tearDown(self):
        pass