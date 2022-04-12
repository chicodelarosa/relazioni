import numpy as np
from relazioni import associations
import unittest

class TestAssociations(unittest.TestCase):
    def test_theils_u(self):
        v1, v2, v3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1]), \
            np.array([-3, 3, -3, 3, -3, 3, -3, 3, -3, 3]), \
                np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])

        self.assertAlmostEqual(0.4946941, associations.theils_u(v1, v3))
        self.assertAlmostEqual(1.0, associations.theils_u(v2, v3))

    def test_matthews_corr(self):
        v1, v2 = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])

        tp = 1
        tn = 1
        fp = 1
        fn = 1

        expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        self.assertAlmostEqual(expected, associations.matthews_corr(v1, v2))

        v1, v2 = np.array(([0] * 94900) + ([1] * 5000) + ([0] * 1) + ([1] * 100)), \
            np.array(([0] * 94900) + ([0] * 5000) + ([1] * 1) + ([1] * 100))

        tp = 100
        tn = 94900
        fp = 5000
        fn = 1

        expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        self.assertAlmostEqual(expected, associations.matthews_corr(v1, v2))

        v1, v2 = np.array(([0] * 9) + ([1] * 1) + ([0] * 10000) + ([1] * 90000)), \
            np.array(([0] * 9) + ([0] * 1) + ([1] * 10000) + ([1] * 90000))

        tp = 90000
        tn = 9
        fp = 1
        fn = 10000
        
        expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        self.assertAlmostEqual(expected, associations.matthews_corr(v1, v2))

        v1, v2 = np.array(([0] * 1) + ([1] * 10) + ([1] * 90000)), \
            np.array(([0] * 1) + ([0] * 10) + ([1] * 90000))

        tp = 90000
        tn = 1
        fp = 10
        fn = 0
        
        expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        self.assertAlmostEqual(expected, associations.matthews_corr(v1, v2))

    def test_cramers_v(self):
        v1, v2 = np.array(([0] * 7) + ([1] * 9) + ([0] * 12) + ([1] * 8)), \
            np.array(([0] * 7) + ([0] * 9) + ([1] * 12) + ([1] * 8))

        self.assertAlmostEqual(0.16174359, associations.cramers_v(v1, v2))
        
        v1, v2 = np.array(
            ([1] * 6) +
            ([1] * 9) +
            ([2] * 8) +
            ([2] * 5) +
            ([3] * 12) +
            ([3] * 9)
        ), \
        np.array(
            ([1] * 6) + 
            ([2] * 9) + 
            ([1] * 8) + 
            ([2] * 5) + 
            ([1] * 12) + 
            ([2] * 9)
        )

        self.assertAlmostEqual(0.17745303, associations.cramers_v(v1, v2))

    def test_kendalls_corr(self):
        v1, v2 = np.array(range(1, 13)), \
            np.array([1, 2, 3, 5, 4, 7, 6, 8, 10, 9, 11, 12])

        self.assertAlmostEqual(0.90909090, associations.kendalls_corr(v1, v2))

    def test_spearmans_corr(self):
        v1, v2 = np.array([35, 23, 47, 17, 10, 43, 9, 6, 28]), \
            np.array([30, 33, 45, 23, 8, 49, 12, 4, 31])
            
        self.assertAlmostEqual(0.9, associations.spearmans_corr(v1, v2))

    def test_pointbiserial_corr(self):
        v1, v2 = np.array([23, 15, 16, 25, 20, 17, 18, 14, 12, 19, 21, 22, 16, 21, 16, 11, 24, 21, 18, 15, 19, 22, 13, 24]), \
            np.array(([1] * 11) + ([0] * 13))
            
        self.assertAlmostEqual(-0.0554839, associations.pointbiserial_corr(v1, v2))
        
    def test_partial_corr(self):
        v1, v2, v3 = np.array([30, 32, 28, 25, 32, 38, 39, 38, 35, 31]), \
            np.array([2.8, 3.0, 2.9, 2.8, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9]), \
                np.array([500, 550, 450, 400, 600, 650, 700, 550, 650, 550])

        self.assertAlmostEqual(0.735660000, associations.partial_corr(v1, v2, v3))
    
    def test_pearson_corr(self):
        v1, v2 = np.array([6, 8, 10]), np.array([12, 10, 20])

        self.assertAlmostEqual(0.755928946, associations.pearson_corr(v1, v2))


if __name__ == '__main__':
    unittest.main()