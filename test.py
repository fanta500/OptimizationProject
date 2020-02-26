import unittest
import numpy as np
from fractions import Fraction
from simplex import lp_solve, Dictionary, bland, LPResult

"""
    Entering:   Choose the first nonbasic variable with non-negative coefficient.
    Leaving:  Choose the basic variable with lowest index i from the set {\forall i : max(a/b)}
"""


class TestBland(unittest.TestCase):
    # B[l],N[k]
    def test_bland_first_index_positive(self):
        """
                     0,  3,  6,  2
                    14,  2, -6,  2
                     8, -4, -7, -4
                     5, -4, -5, -8

                should choose: [0,0] - entering index = 0 from {3,6,2}, leaving index = 0 from {2/14} subset of max({2/14 (0.14285), -4/8 (-0,5), -4/5 (-0.8)})

                     6, -3/4,   3/4, -1
                    18, -1/2, -19/2,  0
                     2, -1/4,  -7/4, -1
                    -3,   1,     2,  -4
        """

        d = Dictionary(np.array([3, 6, 2]), np.array([[-2, 6, -2], [4, 7, 4], [4, 5, 8]]), np.array([14, 8, 5]))
        n, b = bland(d, 0)
        self.assertEqual(0, n)

    def test_bland_some_index_positive(self):
        """
                0,  -8, -7,  2
                14,  2, -6,  2
                8,  -4, -7, -4
                5,  -4, -5, -8

                should choose: [2, 0] - entering index = 2 from {2}, leaving index = 0 from {2/14} subset of max({2/14 (0.14285), -4/8 (-0,5), -8/5 (-1.6)})
        """

        d = Dictionary(np.array([-8, -7, 2]), np.array([[-2, 6, -2], [4, 7, 4], [4, 5, 8]]), np.array([14, 8, 5]))
        n, b = bland(d, 0)
        self.assertEqual(2, n)

    def test_bland_some_fraction_greatest_positive(self):
        """
                 0,  2,  3,  5
                 8, -3,  2,  0
                10,  0, -2, -3
                 6, -1, -1, -1

                should choose: [0, 1] - entering index = 0 from {2,3,5}, leaving index = 1 from {0/10} subset of max({-3/8 (-0.375), 0/10 (0), -1/6 (0.166..)})
        """

        d = Dictionary(np.array([2, 3, 5]), np.array([[3, -2, 0], [0, 2, -3], [1, 1, 1]]), np.array([8, 10, 6]))
        n, b = bland(d, 0)
        self.assertEqual(0, n)
        self.assertEqual(1, b)

    def test_bland_greatest_fraction_zero_div_zero(self):
        """
                0,  2,  3,  5
                8, -3,  2,  0
                0,  0, -2, -3
                6, -1, -1, -1

                should choose: [0, 1] - entering index = 0 from {2,3,5}, leaving index = 1 from {0/0} subset of max({-3/8 (-0.375), 0/0 (0), -1/6 (-0.166..)})
        """

        d = Dictionary(np.array([2, 3, 5]), np.array([[3, -2, 0], [0, 2, -3], [1, 1, 1]]), np.array([8, 0, 6]))
        n, b = bland(d, 0)
        self.assertEqual(0, n)
        self.assertEqual(1, b)

    def test_bland_multiple_greatest_fractions(self):
        """
                0, 1,  7, -1
                4, 2, -3, -3
                8, 4, -3,  2
                8, 1, -6, -3

                should choose: [0, 0] - entering index = 0 from {1,7}, leaving index = 1 from {2/4, 4/8} subset of max({2/4 (0.5), 4/8 (0.5), 1/8 (0.125)})
        """

        d = Dictionary(np.array([1, 7, -1]), np.array([[-2, 3, 3], [-4, 3, -2], [-1, 6, 3]]), np.array([4, 8, 8]))
        n, b = bland(d, 0)
        self.assertEqual(0, n)
        self.assertEqual(0, b)

    def test_bland_detect_optimal(self):
        """
                112, -35/3, -28, -8/3
                45,  -13/3, -17, -1/3
                5,   -1/3,   0,  -1/3
                8,    -1,   -4,    0

                should choose: [None, ?] - entering index = 0 from {}, leaving index = ? from {} subset of max({})
        """

        d = Dictionary(np.array([-35/3, -28, -8/3]), np.array([[13/3, 17, 1/3], [1/3, 0, 1/3], [1, 4, 0]]), np.array([45, 5, 8]))
        n, b = bland(d, 0)
        self.assertEqual(None, n)

    def test_bland_detect_unbounded(self):
        """
                 0, 4,   5
                14, 4, -12
                 7, 9,  -7

                 should choose: [0, None] - entering index = 0 from {4, 5}, leaving index = None
        """

        d = Dictionary(np.array([4, 5]), np.array([[-4, 12], [-9, 7]]), np.array([14, 7]))
        n, b = bland(d, 0)
        self.assertEqual(None, b)


class TestPivot(unittest.TestCase):
    def test_pivot_positive_a(self):
        """
                     0, 11, -3, 20
                    12, -2,  2,  4
                     5,  4,  3,  0
                     4,  5, -3,  8

                pivot C:2,2, N[1],B[1]

                      5,    15,  -1,  20
                    26/3, -14/3, 2/3,  4
                    -5/3,  -4/3, 1/3,  0
                      9,     9,  -1,   8
                """
        d = Dictionary(np.array([11, -3, 20]), np.array([[2, -2, -4], [-4, -3, 0], [-5, 3, -8]]), np.array([12, 5, 4]))
        Dictionary.pivot(d, 1, 1)
        # self.assertEqual(Fraction(-5), d.value())
        l1 = np.array([Fraction(5), Fraction(15), Fraction(-1), Fraction(20)])
        l2 = np.array([Fraction(26, 3), Fraction(-14, 3), Fraction(2, 3), Fraction(4)])
        l3 = np.array([Fraction(-5, 3), Fraction(-4, 3), Fraction(1, 3), Fraction(0)])
        l4 = np.array([Fraction(9), Fraction(9), Fraction(-1), Fraction(8)])
        # self.assertEqual(np.array([l1, l2, l3, l4]), d.C)
        np.testing.assert_array_equal(d.C, np.array([l1, l2, l3, l4]))

    def test_pivot_negative_a(self):
        """
                 z = 0,| 5, 4, 3 = k||N:[0,2]
                        ^^^^^^^^
                   | 5|, -2, -3, -1
                   |11|, -4, -1, -2
                   | 8|, -3, -4, -2
                   ^^^^
                 l||B:[0:2]

                pivot C:1,1, N[0],B[0]

                    25/2, -5/2, -7/2,  1/2
                     5/2, -1/2, -3/2, -1/2
                      1,    2,    5,    0
                     1/2,  3/2,  1/2, -1/2
                """
        d = Dictionary(np.array([5, 4, 3]), np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]), np.array([5, 11, 8]))
        Dictionary.pivot(d, 0, 0)
        # self.assertEqual(Fraction(-25, 2), d.value())
        l1 = np.array([Fraction(25, 2), Fraction(-5, 2), Fraction(-7, 2), Fraction(1, 2)])
        l2 = np.array([Fraction(5, 2), Fraction(-1, 2), Fraction(-3, 2), Fraction(-1, 2)])
        l3 = np.array([Fraction(1), Fraction(2), Fraction(5), Fraction(0)])
        l4 = np.array([Fraction(1, 2), Fraction(3, 2), Fraction(1, 2), Fraction(-1, 2)])
        # self.assertEqual(np.array([l1, l2, l3, l4]), d.C)
        np.testing.assert_array_equal(d.C, np.array([l1, l2, l3, l4]))


class TestExample1(unittest.TestCase):
    def setUp(self):
        self.c = np.array([5, 4, 3])
        self.A = np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]])
        self.b = np.array([5, 11, 8])

    def test_solve(self):
        res, D = lp_solve(self.c, self.A, self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float(self):
        res, D = lp_solve(self.c, self.A, self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
