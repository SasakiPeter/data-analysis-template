import os
import sys
import unittest
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import inverse_cor  # NOQA


class Test(unittest.TestCase):
    def test_example(self):
        expected = np.array([1.0, 0.0, 0.0, 1.0]).reshape(2, 2)
        actual = inverse_cor(np.array([1, 0, 0, 2]).reshape(2, 2))
        np.testing.assert_array_almost_equal(expected, actual)


if __name__ == '__main__':
    unittest.main(verbosity=2)
