"""
DOC
"""
import numpy as np
from simplex import primal_simplex_tableau


def phase_one(A, b):
    """
    Function to compute the feasible starting basis of a linear optimization problem using phase one of the two phase method.
    Returns a optimal tableau for the aritificial LP-problem.
    #### Assumptions
    * The provided LP-problem is the primal problem.
    * The provided LP-problem is on standard form.
    (min c^T·x st. Ax=b, x≥0)
    * The provided basis is dual feasible.
    #### Parameters
    1. A: mxn list
            * A list of the coefficients of the constraints in the primal problem
    2. b: 1xm list
            * A list of the RHS of the maximization problem.

    #### Example usage
    A = [[1,2,3],
        [3,6,9]]

    b= [1,2]

    optimal_aritificial_tableau = phase_one(A,b,c,basis_var)
    """
    A = np.hstack([np.array(A), np.identity(np.array(A).shape[0])])
    basis_var = [A.shape[1] - (i + 1) for i in range(A.shape[0])]
    basis_var.sort()
    c = [1 if i in basis_var else 0 for i in range(A.shape[1])]
    return primal_simplex_tableau(A, b, c, basis_var)
