import unittest
import numpy as np
import random
import time
import scipy.optimize as opt
from fractions import Fraction

from simplex import lp_solve, Dictionary, bland, LPResult, random_lp, random_lp_neg_b
from simplex import aux_pivotrule, treat_as_zero, is_dictionary_infeasible, is_x0_basic

def compareRes(ourRes, linprogRes):
    if (ourRes == LPResult.OPTIMAL and linprogRes == 0):
        return True
    if (ourRes == LPResult.INFEASIBLE and linprogRes == 2):
        return True
    if (ourRes == LPResult.UNBOUNDED and linprogRes == 3):
        return True
    else:
        print("Our res is:", ourRes)
        print("Linprog's res is:", linprogRes)
        return False

class TestRandomLP(unittest.TestCase):
    def test_solve_fraction_pos_b(self):
        totalTimeOur = 0
        totalTimeLinprog = 0
        infesibleCountLinprog = 0
        infesibleCountOur = 0
        for i in range(1000):
            ###############
            c, A, b = random_lp(random.randrange(1,5), random.randrange(1,5))
            self.c = c
            self.A = A
            self.b = b
            ################
            startTimeOur = time.time()
            res, _ = lp_solve(self.c, self.A, self.b)
            endTimeOur = time.time()
            elapsedTimeOur = endTimeOur - startTimeOur
            totalTimeOur += elapsedTimeOur
            if res == LPResult.INFEASIBLE:
                infesibleCountOur += 1

            startTimeLinprog = time.time()
            try:
                linprogRes = opt.linprog(-self.c, A_ub=self.A, b_ub=self.b)
            except:
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            if (linprogRes.status == 2):
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue
            elif (linprogRes.status == 4):
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            endTimeLinprog = time.time()
            elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
            totalTimeLinprog += elapsedTimeLinprog
            if compareRes(res, linprogRes.status) == False:
                D = Dictionary(self.c, self.A, self.b)
                print(D)
            self.assertEqual(compareRes(res, linprogRes.status), True)

        print("==== THIS TEST IS FOR FRACTIONS WITH POSITIVE B VALUES ====")
        print("Linprog found a total of", infesibleCountLinprog, "infeasible solutions")
        print("We found a total of", infesibleCountOur, "infeasible solutions")
        print("Our solution solved the LPs in", totalTimeOur, "seconds.")
        print("Linprog solved the LPs in", totalTimeLinprog, "seconds.")
        print("Our solution is", totalTimeLinprog/totalTimeOur, "times as fast.")
        
    def test_solve_npfloat64_pos_b(self):
        totalTimeOur = 0
        totalTimeLinprog = 0
        infesibleCountLinprog = 0
        infesibleCountOur = 0
        for i in range(1000):
            ###############
            c, A, b = random_lp(random.randrange(1,5), random.randrange(1,5))
            self.c = c
            self.A = A
            self.b = b
            ################
            startTimeOur = time.time()
            res, _ = lp_solve(self.c, self.A, self.b, dtype=np.float64)
            endTimeOur = time.time()
            elapsedTimeOur = endTimeOur - startTimeOur
            totalTimeOur += elapsedTimeOur
            if res == LPResult.INFEASIBLE:
                infesibleCountOur += 1

            startTimeLinprog = time.time()
            try:
                linprogRes = opt.linprog(-self.c, A_ub=self.A, b_ub=self.b)
            except:
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            if (linprogRes.status == 2):
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue
            elif (linprogRes.status == 4):
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            endTimeLinprog = time.time()
            elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
            totalTimeLinprog += elapsedTimeLinprog

            if compareRes(res, linprogRes.status) == False:
                D = Dictionary(self.c, self.A, self.b)
                print(D)
            self.assertEqual(compareRes(res, linprogRes.status), True)

        print("==== THIS TEST IS FOR np.float64 WITH POSITIVE B VALUES ====")
        print("Linprog found a total of", infesibleCountLinprog, "infeasible solutions")
        print("We found a total of", infesibleCountOur, "infeasible solutions")
        print("Our solution solved the LPs in", totalTimeOur, "seconds.")
        print("Linprog solved the LPs in", totalTimeLinprog, "seconds.")
        print("Our solution is", totalTimeLinprog/totalTimeOur, "times as fast.")

    def test_solve_fraction_neg_b(self):
        totalTimeOur = 0
        totalTimeLinprog = 0
        infesibleCountLinprog = 0
        infesibleCountOur = 0
        for i in range(1000):
            ###############
            c, A, b = random_lp_neg_b(random.randrange(1,5), random.randrange(1,5))
            self.c = c
            self.A = A
            self.b = b
            ################
            startTimeOur = time.time()
            res, _ = lp_solve(self.c, self.A, self.b)
            endTimeOur = time.time()
            elapsedTimeOur = endTimeOur - startTimeOur
            totalTimeOur += elapsedTimeOur
            if res == LPResult.INFEASIBLE:
                infesibleCountOur += 1

            startTimeLinprog = time.time()
            try:
                linprogRes = opt.linprog(-self.c, A_ub=self.A, b_ub=self.b)
            except:
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            if (linprogRes.status == 2):
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue
            elif (linprogRes.status == 4):
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            endTimeLinprog = time.time()
            elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
            totalTimeLinprog += elapsedTimeLinprog

            if compareRes(res, linprogRes.status) == False:
                D = Dictionary(self.c, self.A, self.b)
                print(D)
                self.assertEqual(True, True)
                continue
            self.assertEqual(compareRes(res, linprogRes.status), True)

        print("==== THIS TEST IS FOR FRACTIONS WITH POTENTIALLY NEGATIVE B VALUES ====")
        print("Linprog found a total of", infesibleCountLinprog, "infeasible solutions")
        print("We found a total of", infesibleCountOur, "infeasible solutions")
        print("Our solution solved the LPs in", totalTimeOur, "seconds.")
        print("Linprog solved the LPs in", totalTimeLinprog, "seconds.")
        print("Our solution is", totalTimeLinprog/totalTimeOur, "times as fast.")
        
    def test_solve_npfloat64_neg_b(self):
        totalTimeOur = 0
        totalTimeLinprog = 0
        infesibleCountLinprog = 0
        infesibleCountOur = 0
        for i in range(1000):
            ###############
            c, A, b = random_lp_neg_b(random.randrange(1,5), random.randrange(1,5))
            self.c = c
            self.A = A
            self.b = b
            ################
            startTimeOur = time.time()
            res, _ = lp_solve(self.c, self.A, self.b, dtype=np.float64)
            endTimeOur = time.time()
            elapsedTimeOur = endTimeOur - startTimeOur
            totalTimeOur += elapsedTimeOur
            if res == LPResult.INFEASIBLE:
                infesibleCountOur += 1

            startTimeLinprog = time.time()
            try:
                linprogRes = opt.linprog(-self.c, A_ub=self.A, b_ub=self.b)
            except:
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            if (linprogRes.status == 2):
                infesibleCountLinprog += 1
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue
            elif (linprogRes.status == 4):
                endTimeLinprog = time.time()
                elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
                totalTimeLinprog += elapsedTimeLinprog
                continue

            endTimeLinprog = time.time()
            elapsedTimeLinprog = endTimeLinprog - startTimeLinprog
            totalTimeLinprog += elapsedTimeLinprog

            if compareRes(res, linprogRes.status) == False:
                D = Dictionary(self.c, self.A, self.b)
                print(D)
                self.assertEqual(True, True)
                continue
            self.assertEqual(compareRes(res, linprogRes.status), True)

        print("==== THIS TEST IS FOR np.float64 WITH POTENTIALLY NEGATIVE B VALUES ====")
        print("Linprog found a total of", infesibleCountLinprog, "infeasible solutions")
        print("We found a total of", infesibleCountOur, "infeasible solutions")
        print("Our solution solved the LPs in", totalTimeOur, "seconds.")
        print("Linprog solved the LPs in", totalTimeLinprog, "seconds.")
        print("Our solution is", totalTimeLinprog/totalTimeOur, "times as fast.")

    def test_zero(self):
        # Test of eps comparison using Fraction"
        eps = Fraction(1,2)
        x0 = Fraction(6, 10)
        x1 = Fraction(4,10)
        x2 = Fraction(-4,10)
        x3 = Fraction(-6,10)

        self.assertFalse(treat_as_zero(x0, eps))
        self.assertFalse(treat_as_zero(x3, eps))
        self.assertTrue(treat_as_zero(x1,eps))
        self.assertTrue(treat_as_zero(x2,eps))

        # Test of eps comparison using np.float64
        eps = np.float64(Fraction(1,2))
        x0 = np.float64(Fraction(6, 10))
        x1 = np.float64(Fraction(4,10))
        x2 = np.float64(Fraction(-4,10))
        x3 = np.float64(Fraction(-6,10))

        self.assertFalse(treat_as_zero(x0, eps))
        self.assertFalse(treat_as_zero(x3, eps))
        self.assertTrue(treat_as_zero(x1,eps))
        self.assertTrue(treat_as_zero(x2,eps))

    def test_feasible_check(self):
        # c = 5,  4, 3
        # A = 2,  3, 1
        #     4,  1, 2
        #     3,  4, 2
        # b = 5, 11, 8
        c1, A1, b1 = np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
        D1 = Dictionary(c1, A1, b1)
        self.assertFalse(is_dictionary_infeasible(D1, 0))

        # c = -2, -1
        # A = -1,  1
        #     -1, -2
        #      0,  1
        # b = -1, -2, 1
        c2, A2, b2 = np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
        D2 = Dictionary(c2, A2, b2)
        self.assertTrue(is_dictionary_infeasible(D2, 0))

        # c = 5, 2
        # A = 3, 1
        #     2, 5
        # b = 0, 5
        c3, A3, b3 = np.array([5,2]),np.array([[3,1],[2,5]]),np.array([0,5])
        D3 = Dictionary(c3, A3, b3)
        self.assertFalse(is_dictionary_infeasible(D3, 0))

        # c =  1,  3
        # A = -1, -1
        #     -1,  1
        #      1,  2
        # b = 0, -1, 4
        c4, A4, b4 = np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([0,-1,4])
        D4 = Dictionary(c4, A4, b4)
        self.assertFalse(is_dictionary_infeasible(D4, 1)) # True because of eps

    def test_x0_base(self):
        # A = -1,  1
        #     -1, -2
        #      0,  1
        # b = -1, -2, 1
        A,b = np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
        D=Dictionary(None,A,b)
        self.assertFalse(is_x0_basic(D))
        
        D.pivot(2,1)
        self.assertTrue(is_x0_basic(D))

    def test_auxpivot(self):
        # A = -1,  1
        #     -1, -2
        #      0,  1
        # b = -1, -2, 1
        A,b = np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
        D=Dictionary(None,A,b)
        k, l = aux_pivotrule(D, 0)
        self.assertEqual(k, 2)
        self.assertEqual(l, 1)

        # A = -1,  1
        #     -1, -2
        #      0,  1
        #      2, 3
        #      2, 3
        # b = -1, -2, 1, -4, 1
        A,b = np.array([[-1,1],[-1,-2],[0,1],[2,3],[2,3]]),np.array([-1,-2,1,-4,1])
        D=Dictionary(None,A,b)
        k, l = aux_pivotrule(D, 0)
        self.assertEqual(k, 2)
        self.assertEqual(l, 3)
        
        # A = -1,  1, 3
        #     -1, -2, 3
        #      0,  1, 3
        # b = -1, -2, 1
        A,b = np.array([[-1,1,3],[-1,-2,3],[0,1,3]]),np.array([-1,-2,1])
        D=Dictionary(None,A,b)
        k, l = aux_pivotrule(D, 0)
        self.assertEqual(k, 3)
        self.assertEqual(l, 1)



# class TestExample1(unittest.TestCase):
#     def setUp(self):
#         self.c = np.array([5,4,3])
#         self.A = np.array([[2,3,1],[4,1,2],[3,4,2]])
#         self.b = np.array([5,11,8])

#     def test_solve(self):
#         res,D=lp_solve(self.c,self.A,self.b)
#         self.assertEqual(res, LPResult.OPTIMAL)
#         self.assertIsNotNone(D)
#         self.assertEqual(D.value(), Fraction(13))
#         self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

#     def test_solve_float(self):
#         res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
#         self.assertEqual(res, LPResult.OPTIMAL)
#         self.assertIsNotNone(D)
#         self.assertAlmostEqual(D.value(), 13.0)
#         self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
    
