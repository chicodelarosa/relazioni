import numpy as np
import relationships
import unittest

class TestAssociations(unittest.TestCase):
    def test_pearson(self):
        v1, v2 = np.array([6, 8, 10]), np.array([12, 10, 20])

        self.assertAlmostEqual(0.755928946, relationships.pearson_corr(v1, v2))

    def test_partial_corr(self):
        v1, v2 = np.array([2, 4, 15, 20]), np.array([1, 2, 3, 4])

        print(relationships.partial_corr(v1, v2))
        self.assertAlmostEqual(0.970, relationships.partial_corr(v1, v2))

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()