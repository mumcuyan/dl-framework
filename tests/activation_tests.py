import unittest

from unittest import TestCase

class ActivationTests(unittest.TestCase):

    def __init__(self):
        super(ActivationTests, self).__init__()


    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestDatabaseAdapter(BaseTest, TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db_adapter = DatabaseAdapter(self._session)

        self.start_params = {
            "id": "-1",
            "username": "a",
            "fullName": "b",
            "gender" : "x",
            "isPublic": "False",
            "totalFollowers": "2",
        }
        self.test_instauser = Instauser(**self.start_params)

        self.correct_updates = {
            "username": "sd",
            "fullName": "c",
            "gender": "p",
        }

        self.null_updates = {
            "username": None
        }

        self.wrong_params = {
            "wrong_param": "wrong value"
        }

    def setUp(self):
        """
        This method is run once before _each_ test method is executed
        """
        super().setUp()
        self.db_adapter.insert(self.test_instauser)

    def test_01_get(self):
        get_user = self.db_adapter.get(Instauser, self.test_instauser.id)
        assert get_user.id == self.test_instauser.id

    def test_01_wrong_get(self):
        with self.assertRaises(NotFound):
            self.db_adapter.get(Instauser, "-2")

    def test_02_update_correct(self):
        self.db_adapter.update(Instauser, self.test_instauser.id, self.correct_updates)
        updated_user = self.db_adapter.get(Instauser, self.test_instauser.id)

        assert updated_user.username == self.correct_updates["username"]
        assert updated_user.fullName == self.correct_updates["fullName"]
        assert updated_user.gender == self.correct_updates["gender"]

    def test_02_update_wrong_id(self):
        with self.assertRaises(NotFound):
            self.db_adapter.update(Instauser, "-2", self.correct_updates)

    def test_02_update_wrong_param(self):

        with self.assertRaises(WrongColumnName):
            self.db_adapter.update(Instauser, self.test_instauser.id, self.wrong_params)

    def test_02_update_notnull2null(self):

        with self.assertRaises(ColumnNullException):
            self.db_adapter.update(Instauser, self.test_instauser.id, self.null_updates)

    def test_03_insert_duplicate(self):
        self.db_adapter.insert(self.test_instauser)
        print("insert1")
        self.db_adapter.insert(self.test_instauser)
        print("insert1")
        self.db_adapter._session.commit()
        self.db_adapter.insert(self.test_instauser)
        print("insert1")
        self.db_adapter.insert(self.test_instauser)
        print("insert1")
        self.db_adapter.insert(self.test_instauser)
        print("insert1")

    def test_04_insert_null_id(self):

        with self.assertRaises(ColumnNullException):
            self.db_adapter.insert(Instauser(id=None))

    def tearDown(self):
        super().setUp()
        self.db_adapter.delete(self.test_instauser)


# add new columns
# update columns
# raise exceptions with assert
# ekstra insert
# assertRaises

# Exceptions
# update -> not found
# get -> not found
# duplicate error
# delete yoktu ? not found
# insert -> duplicate
# insert -> not null -> id

