from unittest import TestCase
from modules import Sequential, Linear


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

    def initialization_test_1(self):
        model = Sequential(
            [
                Linear(input_size=2, out=22, activation='tanh')
            ]
        )
        pass

    def module_add_test(self):
        model = Sequential()
        model.add(Linear(out=4, input_size=2, activation='relu'))
        model.add(Linear(out=2, input_size=4, activation='relu'))

        # Error following
        model.add(Linear(out=2, input_size=3, activation='relu'))

    def activation_test(self):
        pass

    def tearDown(self):
        super().tearDown()
