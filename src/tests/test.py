import unittest


class Test(unittest.TestCase):
    def setUp(self):
        self.hoge = 'hoge'

    def test_example(self):
        expected = 'hoge'
        actual = 'hoge'
        self.assertEqual(expected, actual)

    def tearDown(self):
        del self.hoge


if __name__ == '__main__':
    unittest.main(verbosity=2)
