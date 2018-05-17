from unittest import  TestCase
from utils import label2one_hot
from utils.generate_data import generate_data
from examples.networks import get_network, get_network_ce_1, get_network_ce_2, \
    get_network_ce_3, get_network_ce_4, \
    get_network_mse_1, get_network_mse_2, get_network_mse_3


class NetworkTest(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """  categorical dataset """
        self.x_train_cat, y_train_label = generate_data(num_of_points=1000)
        self.y_train_cat = label2one_hot(y_train_label, val=0)  # convert labels to 1-hot encoding

        """ mse dataset """
        self.x_train_mse, y_train_label = generate_data(num_of_points=1000)
        self.y_train_mse = label2one_hot(y_train_label, val=-1)  # convert

        self.ce_funcs = [get_network, get_network_ce_1, get_network_ce_2, get_network_ce_3, get_network_ce_4]
        self.mse_funcs = [get_network, get_network_mse_1, get_network_mse_2, get_network_mse_3]

    def test_ce_funcs(self):
        raised = False
        try:
            for idx, network_func in enumerate(self.ce_funcs):
                print("{} is being tested ...".format(network_func.__name__))
                network_func(self.x_train_cat, self.y_train_cat)
        except Exception as e:
            raised = True

        self.assertFalse(raised, 'Exception raised @ @test_ce_funcs')

    def test_mse_funcs(self):
        raised = False
        try:
            for idx, network_func in enumerate(self.mse_funcs):
                print("{} is being tested ...".format(network_func.__name__))
                network_func(self.x_train_mse, self.y_train_mse)
        except Exception as e:
            print(e)
            raised = True

        self.assertFalse(raised, 'Exception raised @ test_mse_funcs')

    def tearDown(self):
        pass