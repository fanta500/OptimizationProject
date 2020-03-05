import numpy as np
import math
import sys
from decimal import Decimal
from fractions import Fraction
from enum import Enum

# c = 5,  4, 3
# A = 2,  3, 1
#     4,  1, 2
#     3,  4, 2
# b = 5, 11, 8
def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])

# c = -2, -1
# A = -1,  1
#     -1, -2
#      0,  1
# b = -1, -2, 1
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])

# c = 5, 2
# A = 3, 1
#     2, 5
# b = 7, 5
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])

# c =  1,  3
# A = -1, -1
#     -1,  1
#      1,  2
# b = -3, -1, 4
def exercise2_5(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])

# c =  1,  3
# A = -1, -1
#     -1,  1
#      1,  2
# b = -3, -1, 2
def exercise2_6(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])

# c =  1,  3
# A = -1, -1
#     -1,  1
#     -1,  2
# b = -3, -1, 2
def exercise2_7(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def random_lp(n,m,sigma=10): return np.round(sigma*np.random.randn(n)), np.round(sigma*np.random.randn(m,n)), np.round(sigma*np.abs(np.random.randn(m)))
def random_lp_neg_b(n,m,sigma=10): return np.round(sigma*np.random.randn(n)), np.round(sigma*np.random.randn(m,n)), np.round(sigma*np.random.randn(m))


class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.
    
    def __init__(self,c,A,b,dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A' 
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m,n = A.shape
        self.dtype=dtype
        if dtype == int:
            self.lastpivot=1
        if dtype in [int,Fraction]:
            dtype=object
            if c is not None:
                c=np.array(c,np.object)
            A=np.array(A,np.object)
            b=np.array(b,np.object)
        self.C = np.empty([m+1,n+1+(c is None)],dtype=dtype)
        self.C[0,0]=self.dtype(0)
        if c is None:
            self.C[0,1:]=self.dtype(0)
            self.C[0,n+1]=self.dtype(-1)
            self.C[1:,n+1]=self.dtype(1)
        else:
            for j in range(0,n):
                self.C[0,j+1]=self.dtype(c[j])
        for i in range(0,m):
            self.C[i+1,0]=self.dtype(b[i])
            for j in range(0,n):
                self.C[i+1,j+1]=self.dtype(-A[i,j])
        self.N = np.array(range(1,n+1+(c is None)))
        self.B = np.array(range(n+1+(c is None),n+1+(c is None)+m))
        self.varnames=np.empty(n+1+(c is None)+m,dtype=object)
        self.varnames[0]='z'
        for i in range(1,n+1):
            self.varnames[i]='x{}'.format(i)
        if c is None:
            self.varnames[n+1]='x0'
        for i in range(n+1,n+m+1):
            self.varnames[i+(c is None)]='x{}'.format(i)

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m,n = self.C.shape
        varlen = len(max(self.varnames,key=len))
        coeflen = 0
        for i in range(0,m):
            coeflen=max(coeflen,len(str(self.C[i,0])))
            for j in range(1,n):
                coeflen=max(coeflen,len(str(abs(self.C[i,j]))))
        tmp=[]
        if self.dtype==int and self.lastpivot!=1:
            tmp.append(str(self.lastpivot))
            tmp.append('*')
        tmp.append('{} = '.format(self.varnames[0]).rjust(varlen+3))
        tmp.append(str(self.C[0,0]).rjust(coeflen))
        for j in range(0,n-1):
            tmp.append(' + ' if self.C[0,j+1]>0 else ' - ')
            tmp.append(str(abs(self.C[0,j+1])).rjust(coeflen))
            tmp.append('*')
            tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0,m-1):
            tmp.append('\n')
            if self.dtype==int and self.lastpivot!=1:
                tmp.append(str(self.lastpivot))
                tmp.append('*')
            tmp.append('{} = '.format(self.varnames[self.B[i]]).rjust(varlen+3))
            tmp.append(str(self.C[i+1,0]).rjust(coeflen))
            for j in range(0,n-1):
                tmp.append(' + ' if self.C[i+1,j+1]>0 else ' - ')
                tmp.append(str(abs(self.C[i+1,j+1])).rjust(coeflen))
                tmp.append('*')
                tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        return ''.join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m,n = self.C.shape
        if self.dtype==int:
            x_dtype=Fraction
        else:
            x_dtype=self.dtype
        x = np.empty(n-1,x_dtype)
        x[:] = x_dtype(0)
        for i in range (0,m-1):
            if self.B[i]<n:
                if self.dtype==int:
                    x[self.B[i]-1]=Fraction(self.C[i+1,0],self.lastpivot)
                else:
                    x[self.B[i]-1]=self.C[i+1,0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype==int:
            return Fraction(self.C[0, 0], self.lastpivot)
        else:
            return self.C[0, 0]

    def pivot(self, k, l):  
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        # save pivot coefficient
        a = self.C[l+1,k+1] # Coefficient to divide leaving equation by when solving for entering?

        xEntering = self.N[k]
        xLeaving = self.B[l]
        # print("The dictionary is")
        # print(self)
        # print("Entering var is", k)
        # print("Leaving var is", l)
        
        # Solve xLeaving equation for xEntering
        row = -np.copy(self.C[l+1]) #row of leaving var
        row = np.divide(row, a) #div all coefs by a
        row[k+1] = np.divide(1, a) #set the leaving var to -1/a
        #print("the row of the leaving var is", row)
        self.C[l+1] = row
        # Update C
        for i in range(len(self.C)):
            if i == l+1: #skip the row we already modified
                continue
            else:
                enteringCoef = np.copy(self.C[i, k+1]) #coefficient of the entering var in the equation for all other equations (NOT in leaving var equation)
                self.C[i] = self.C[i] + enteringCoef * row #all coefs except leaving var are set correctly
                self.C[i, k+1] = enteringCoef * self.C[l+1, k+1] #sets the coefs for the leaving vars correctly
            # print("The dictionary is")
            # print(self)
                
        # Update N
        self.N[k] = xLeaving
        # Update B
        self.B[l] = xEntering
        

class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3

def treat_as_zero(x, eps):
    if -eps <= x and x <= eps:
        return True
    else:
        return False

def compute_when_a_b_is_zero(D, eps, k):
    '''
        This method takes the dictionary, eps and k
        to filter through the a and b columns and treat any 0-value properly
    '''
    enteringVarColumn = D.C[1:, k+1]
    bValueColumn = D.C[1:, 0]
    BAarr = np.column_stack((bValueColumn, enteringVarColumn)) #glue the b values with the a values of the entering var
    for i in range(len(BAarr)):
        #respect epsilon again here
        if (treat_as_zero(BAarr[i, 0], eps)):
            BAarr[i, 0] = 0
        if (treat_as_zero(BAarr[i, 1], eps)):
            BAarr[i, 1] = 0
        #makes sure that the correct corner cases are treated properly
        if (BAarr[i, 0] == 0 and (BAarr[i, 1] == 0)): #both a and b are 0, the resulting fraction of -a/b must be 0 for this special case. Report unbounded
            BAarr[i, 1] = 0 #set a to be 0
            BAarr[i, 0] = 1 #set b to be 1. Ends up being 0/1 which is 0
        elif BAarr[i, 0] == 0:
            signOf_a = np.sign(BAarr[i, 1]) #above case handles a = 0, so the sign can never return 0
            BAarr[i, 1] = signOf_a * sys.float_info.max
            BAarr[i, 0] = 1
    return BAarr

def bland(D,eps):
    # Assumes a feasible dictionary D and finds entering and leaving
    # variables according to Bland's rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
       
    k = l = None

    obj = D.C[0, 1:] #this selects the first row and all columns except the first one
    try:
        #print(np.where(obj > eps))
        lowestIndexWithPosCoef = np.where(obj > eps)[0][0] #leftmost column with coef > 0
    except:
        return None, None
    k = lowestIndexWithPosCoef

    if checkUnbounded(D, k, eps):
        return k, None

    BAarr = compute_when_a_b_is_zero(D, eps, k) 

    # apparently we should use highest ratio of -a/b instead of lowest of b/a. Section 2.4 in Vanderbei 
    highestRatio = np.sort(np.divide(-BAarr[:, 1], BAarr[:, 0]))[len(BAarr)-1]
    if highestRatio <= eps:
        return k, None
    indexInB = None
    for i in range(len(BAarr)):
        if highestRatio == np.divide(-BAarr[i, 1], BAarr[i, 0]):
            indexInB = i  
    l = indexInB

    return k,l

def checkUnbounded(D, k, eps):
    for i in range(len(D.B)):
        if D.C[i+1, k+1] < eps:
            return False
    return True

def largest_coefficient(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k = l = None

    obj = D.C[0, 1:] #this selects the first row and all columns except the first one
    largestCoef = np.sort(obj)[len(obj)-1]
    if largestCoef <= eps: #if the largest coef is smaller or equal to eps, return optimal
        return None, None
    indexInN = np.where(obj == largestCoef)[0][0]
    k = indexInN

    if checkUnbounded(D, k, eps):
        return k, None

    enteringVarColumn = D.C[1:, k+1]
    bValueColumn = D.C[1:, 0]
    BAarr = np.column_stack((bValueColumn, enteringVarColumn)) #glue the b values with the a values of the entering var
    for i in range(len(BAarr)):
        #respect epsilon again here
        if (treat_as_zero(BAarr[i, 0], eps)):
            BAarr[i, 0] = Fraction(0,1)
        if (treat_as_zero(BAarr[i, 1], eps)):
            BAarr[i, 1] = Fraction(0,1)
        #makes sure that the correct corner cases are treated properly
        if (BAarr[i, 0] == Fraction(0,1) and (BAarr[i, 1] == Fraction(0,1))): #both a and b are 0, the resulting fraction of -a/b must be 0 for this special case
            BAarr[i, 1] = Fraction(0,1) #set a to be 0
            BAarr[i, 0] = Fraction(1,1) #set b to be 1. Ends up being -0/1 which is 0
        elif BAarr[i, 0] == Fraction(0,1):
            signOf_a = np.sign(BAarr[i, 1]) #above case handles a = 0, so the sign can never return 0
            BAarr[i, 1] = signOf_a * Fraction(sys.float_info.max).limit_denominator()
            BAarr[i, 0] = Fraction(1,1)

    # apparently we should use highest ratio of a/b instead of lowest of b/a. Section 2.4 in Vanderbei
    highestRatio = np.sort(np.divide(-BAarr[:, 1], BAarr[:, 0]))[len(BAarr)-1]
    indexInB = None
    for i in range(len(BAarr)):
        if highestRatio == np.divide(-BAarr[i, 1], BAarr[i, 0]):
            indexInB = i  
    l = indexInB

    return k,l

def largest_increase(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k=l=None
    # TODO
    return k,l

def is_dictionary_infeasible(D, eps):
    # Dict. is feasible if all b's are nonnegative. Ie C[i,0] >= 0 (with eps).
    for i in range(len(D.B)):
        if D.C[i+1, 0] < -eps:
            return True
    return False

def get_x0_index(D):
    # The index of x0, ie the value in D.N and D.B that corresponds to x0
    _, w = D.C.shape
    x0_index = w-1
    return x0_index

def aux_pivotrule(D, eps):
    # Choose pivot variables for first aux. dictionary pivot. 
    # Select x0 as entering and leaving variable as the one with minimal (most negative) b. (Lecture 2, slide 40)
    
    # x0 seems to be located rightmost, but not specified for lp_solve so make sure
    x0_index = get_x0_index(D) 
    N_pos = np.where(D.N == x0_index)[0][0]
    k = N_pos

    b_col = D.C[:,0] # value of objective function should be 0, so no need to remove
    minimal_b = np.sort(b_col)[0] 
    index_of_minimal = np.where(b_col == minimal_b)[0][0] # Is this also okay for floats?
    l = index_of_minimal - 1

    return k, l

def is_x0_basic(D):
    # Check if x0 is in basis
    x0_index = get_x0_index(D)
    return (x0_index in D.B)


def express_objective(D_origin, D_aux):
    _, width = D_aux.C.shape
    D_aux.C[0] = np.zeros(width, D_origin.dtype) # ensure all coefs in aux obj are 0
    obj_origin = D_origin.C[0]

    for i in range(1, width): # variables 1 to n are the original variables
        # i is the index of the variable
        origin_factor = obj_origin[i]
        #print("The factor for x", i, " is ", origin_factor)

        aux_present = np.where(D_aux.B == i)[0]
        #print("The search for the variable in aux gave: ", aux_present)
        should_obj_change = (len(aux_present) == 1)
        #print("Should obj change for x", i, ": ", should_obj_change)
        if should_obj_change:
            nonbase_row_index = aux_present[0] + 1
            nonbase_row = D_aux.C[nonbase_row_index]
            #print("The additive row for x", i, "is:", origin_factor * nonbase_row)
            D_aux.C[0] = D_aux.C[0] + origin_factor * nonbase_row
        else:
            new_index = np.where(D_aux.N == i)[0][0] + 1
            #print("Adding", origin_factor, "to ", D_aux.C[0,new_index])
            D_aux.C[0, new_index] += origin_factor
        #print("The dict is now\n", D_aux)
    
    #print("The new feasible dict is\n", D_aux)
    return D_aux


def lp_solve(c,A,b,dtype=Fraction,eps=0,pivotrule=lambda D: bland(D,eps=0),verbose=False):
    # Simplex algorithm
    #    
    # Input is LP in standard form given by vectors and matrices
    # c,A,b.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0.
    #
    # pivotrule is a rule used for pivoting. Cycling is prevented by
    # switching to Bland's rule as needed.
    #
    # If verbose is True it outputs possible useful information about
    # the execution, e.g. the sequence of pivot operations
    # performed. Nothing is required.
    #
    # If LP is infeasible the return value is LPResult.INFEASIBLE,None
    #
    # If LP is unbounded the return value is LPResult.UNBOUNDED,None
    #
    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,D, where D is an optimal dictionary.

    D = Dictionary(c, A, b)
    
    # print("The original dict is")
    # print(D)
    if is_dictionary_infeasible(D, eps):
        #create aux dict. Using none makes it for us
        D_aux = Dictionary(None, A, b)
        # print("The aux dict is")
        # print(D_aux)
        #make initial pivot of x0 and the most "infeasible" (largest negative value) basic var
        k_aux, l_aux = aux_pivotrule(D_aux, eps)
        D_aux.pivot(k_aux, l_aux)
        # print("The aux dict is")
        # print(D_aux)
        while True: 
            #make pivots in the now feasible dict
            k_aux, l_aux = pivotrule(D_aux)
            #print("Index of entering is", k_aux, "and index of leaving is", l_aux)
            if k_aux is None: #if the entering var is none, then the aux dict is optimal
                break
            elif l_aux is None:
                return LPResult.UNBOUNDED, None
            D_aux.pivot(k_aux, l_aux)
            # print("The aux dict is")
            # print(D_aux)
        objValueAux = D_aux.C[0,0]
        # print("The value of the objective func is", objValueAux)
        if objValueAux < -eps: #if the optimal aux dict has optimal solution less than 0, the original LP is infeasible
            return LPResult.INFEASIBLE, None  
        # print("The aux dict is")
        # print(D_aux) 
        if is_x0_basic(D_aux): #if x0 is in the basis, pivot it out
            #print("x0 is basic")
            x0_index = get_x0_index(D_aux) 
            B_pos = np.where(D_aux.B == x0_index)[0][0]
            l = B_pos
            k_ph = np.where(D_aux.C[l+1,1:] is not 0)
            print(k_ph)
            print(D_aux.C)
            D_aux.pivot(len(D_aux.C[0])-2, l) #-2 is because we need to remove the consideration of objective value and column for aux var
            D_aux.C = np.delete(D_aux.C, l+1, axis=1)
            D_aux.N = np.delete(D_aux.N, l)
        else: #if x0 is not in the basis, remove it
            x0_index = get_x0_index(D_aux) 
            N_pos = np.where(D_aux.N == x0_index)[0][0]
            l = N_pos
            D_aux.C = np.delete(D_aux.C, l+1, axis=1) #delete the column that is x0
            D_aux.N = np.delete(D_aux.N, l)

        D = express_objective(D, D_aux)
    
    while True:
        k, l = pivotrule(D)
        # print(k)
        if k is None:
            return LPResult.OPTIMAL, D

        if l is None:
            return LPResult.UNBOUNDED, None 

        D.pivot(k,l)

    return None,None
  
def run_examples():
    # # Example 1
    # c,A,b = example1()
    # D=Dictionary(c,A,b)
    # print('Example 1 with Fraction')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()

    # D=Dictionary(c,A,b,np.float64)
    # print('Example 1 with np.float64')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()

    # # Example 2
    # c,A,b = example2()
    # print('Example 2')
    # print('Auxillary dictionary')
    # D=Dictionary(None,A,b)
    # print(D)
    # print('x0 is entering and x4 leaving:')
    # D.pivot(2,1)
    # print(D)
    # print('x2 is entering and x3 leaving:')
    # D.pivot(1,0)
    # print(D)
    # print('x1 is entering and x0 leaving:')
    # D.pivot(0,1)
    # print(D)
    # print()

    # Solve Example 1 using lp_solve
    c,A,b = example1()
    print('lp_solve Example 1:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

    # Solve Example 2 using lp_solve
    c,A,b = example2()
    print('lp_solve Example 2:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.5 using lp_solve
    c,A,b = exercise2_5()
    print('lp_solve Exercise 2.5:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.6 using lp_solve
    c,A,b = exercise2_6()
    print('lp_solve Exercise 2.6:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.7 using lp_solve
    c,A,b = exercise2_7()
    print('lp_solve Exercise 2.7:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

    # # #Integer pivoting
    # # c,A,b=example1()
    # # D=Dictionary(c,A,b,int)
    # # print('Example 1 with int')
    # # print('Initial dictionary:')
    # # print(D)
    # # print('x1 is entering and x4 leaving:')
    # # D.pivot(0,0)
    # # print(D)
    # # print('x3 is entering and x6 leaving:')
    # # D.pivot(2,2)
    # # print(D)
    # # print()

    # # c,A,b = integer_pivoting_example()
    # # D=Dictionary(c,A,b,int)
    # # print('Integer pivoting example from lecture')
    # # print('Initial dictionary:')
    # # print(D)
    # # print('x1 is entering and x3 leaving:')
    # # D.pivot(0,0)
    # # print(D)
    # # print('x2 is entering and x4 leaving:')
    # # D.pivot(1,1)
    # # print(D)

    # # Solve Exmaple slide 42 lec 2 using lp_solve
    # c = np.array([1,-1,1])
    # A = np.array([np.array([2,-3,1]),np.array([2,-1,2]),np.array([-1,1,-2])])
    # b = np.array([-5,4,-1])
    # print('lp_solve Ex 42 2')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # Solve randomly generated problem
    c = np.array([-5,-9,16,8])
    A = np.array([np.array([0,3,0,20])])
    b = np.array([-13])
    print('lp_solve random generated lp')
    print(Dictionary(c,A,b))
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()

if __name__ == "__main__":
    run_examples()