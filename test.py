import unittest
import numpy as np
from fractions import Fraction
from simplex import lp_solve, Dictionary, bland, LPResult


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
