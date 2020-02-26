import unittest
import numpy as np
import random
import time
import scipy.optimize as opt
from fractions import Fraction

from simplex import lp_solve, Dictionary, bland, LPResult, random_lp

def compareRes(ourRes, linprogRes):
    if (ourRes == LPResult.OPTIMAL and linprogRes == 0):
        return True
    if (ourRes == LPResult.INFEASIBLE and linprogRes == 2):
        return True
    if (ourRes == LPResult.UNBOUNDED and linprogRes == 3):
        return True

class TestRandomLP(unittest.TestCase):
    def setUp(self):
        c, A, b = random_lp(random.randrange(20), random.randrange(20))
        self.c = c
        self.A = A
        self.b = b
        # print(c)
        # print(A)
        # print(b)

    def test_solve(self):
        totalTimeOur = 0
        totalTimeLinprog = 0
        for i in range(1000):
            print("problem number", i+1, "started.")
            ###############
            c, A, b = random_lp(random.randrange(20), random.randrange(20))
            self.c = c
            self.A = A
            self.b = b
            ################
            startTimeLinprog = time.time()
            linprogRes = opt.linprog(c, A, b)
            endTimeLinprog = time.time()
            print("Linprog returns" ,linprogRes.status)
            if (linprogRes.status == 2):
                continue
            elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
            totalTimeLinprog += elapsedTimeLinprog

            startTimeOur = time.time()
            res, _ = lp_solve(self.c, self.A, self.b)
            endTimeOur = time.time()
            print("Our solution returns", res)
            elapsedTimeOur = endTimeOur - startTimeOur
            totalTimeOur += elapsedTimeOur

            self.assertEqual(compareRes(res, linprogRes.status), True)

        print("Our solution solved the LPs in", totalTimeOur, "seconds.")
        print("Linprog solved the LPs in", totalTimeLinprog, "seconds.")
        print("Our solution is", totalTimeOur/totalTimeLinprog, "times as fast.")
        

class TestExample1(unittest.TestCase):
    def setUp(self):
        self.c = np.array([5,4,3])
        self.A = np.array([[2,3,1],[4,1,2],[3,4,2]])
        self.b = np.array([5,11,8])

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])

if __name__ == '__main__':
    unittest.main()
    
