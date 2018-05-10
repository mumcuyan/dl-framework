from unittest import TestCase
from modules import Sequential, Linear
from exceptions import *


class BaseTest:
    pass


class SequentialTest(BaseTest, TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        This method is run once before _each_ test method is executed
        """
        super().setUp()

    def test_init_not_input_size(self):
        with self.assertRaises(InputSizeNotFoundError):
            model = Sequential(
                [
                    Linear(out=22, activation='tanh'), # NO input_size is given
                    Linear(input_size=23, out=22, activation='tanh')
                ]
            )

    def test_init_not_compatible(self):

        with self.assertRaises(NotCompatibleError):
            model = Sequential(
                [
                    Linear(input_size=2, out=22, activation='tanh'),
                    Linear(input_size=23, out=22, activation='tanh')
                    # second layer's input_size is not compatible with previous layer output_size
                ]
            )

    def test_training(self):
        pass

    def test_save_model(self):
        model = Sequential()
        model.add(Linear(input_size=2, out=24, activation='tanh'))
        model.add(Linear(input_size=48, out=2, activation='tanh'))

        pass

    def test_load_model(self):
        model = Sequential()
        model.add(Linear(input_size=2, out=24, activation='tanh'))
        model.add(Linear(input_size=48, out=2, activation='tanh'))

        file_name = "model.h5py"
        model = Sequential.load_from_disk(file_name)

        assert str(model) == str(model)

    def tearDown(self):
        super().tearDown()
