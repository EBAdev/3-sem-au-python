"""
DOC
"""
import numpy as np
import fractions


def tableau_to_latex_table(matrix, convert_decimals_to_fractions=True, caption=""):
    """Converts a NumPy matrix to a LaTeX table.

    Args:
      matrix: A NumPy matrix.
      convert_decimals_to_fractions: A boolean value indicating whether to convert decimals to fractions.

    Returns:
      A LaTeX table string.
    """

    # Create the LaTeX table header.
    table_header = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{"
    for column in range(matrix.shape[1]):
        table_header += "c"

    table_header += "}"

    # Create the LaTeX table body.
    header = ""
    for i in range((matrix.shape[1] - 2)):
        header += "  $x_" + str(i + 1) + "$ & "
    table_body = "$Z$ &" + header + " $RHS$"
    for i, row in enumerate(matrix):
        table_body += "\\\\\n"
        if i == 0:
            table_body += "\\toprule"
        for idx, element in enumerate(row):
            if convert_decimals_to_fractions and isinstance(element, float):
                element = fractions.Fraction(element).limit_denominator(1000000)
            if idx == len(row) - 1:
                table_body += str(element)
            else:
                table_body += str(element) + "&"

    # Create the complete LaTeX table.
    latex_table = (
        table_header
        + "\n"
        + table_body
        + "\\\\\\bottomrule\n\\end{tabular}"
        + "\n\\caption{"
        + caption
        + "}"
        + "\n\\end{table}"
    )

    return latex_table


def np_mat_to_latex_pmatrix(a):
    """
    Returns string formattet as a LaTeX pmatrix
    #### Parameters
    1. a: numpy array
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{pmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{pmatrix}"]
    return "\n".join(rv)


def calculate_reduced_costs(A, A_bar, c_B, c_T, J):
    reduced_costs = []
    for i in range(A.shape[1]):
        if i in J:
            reduced_costs.append(c_T[0][i] - (c_B.T @ A_bar[:, i])[0])
        else:
            reduced_costs.append(0)
    reduced_costs = np.atleast_2d(np.array(reduced_costs)).T
    return reduced_costs


def get_A_basis_matrix(A, B):
    return np.array([A[:, i] for i in B]).T


def get_c_basis_matrix(c_T, B):
    return np.array([c_T[:, i] for i in B])


def calculate_A_bar_matrix(A, A_B):
    return np.array(np.linalg.inv(A_B) @ A)


def get_primal_solution(A, b, B, J):
    A_B = get_A_basis_matrix(A, B)
    primal_solution = [
        0 if x_bar in J else np.linalg.solve(A_B, b)[B.index(x_bar)][0]
        for x_bar in range(A.shape[1])
    ]
    return np.atleast_2d(np.array(primal_solution)).T  # primal solution


def primal_usable(x_bar):
    for x in x_bar:
        if np.isclose(x, 0, rtol=0.000001):
            continue  # fix tolerance on reduced cost being almost pos/neg
        if x < 0:
            return False
    return True


def dual_useable(reduced_costs):
    for c in reduced_costs:
        if np.isclose(c, 0, rtol=0.000001):
            continue
        if c < 0:
            return False
    return True


def get_z_vector(A_bar):
    z = []  # make list comprehension
    for i in range(A_bar.shape[0] + 1):
        if i == 0:
            z.append(-1)
        else:
            z.append(0)
    z = np.atleast_2d(np.array(z)).T
    return z


def create_tableau(z, reduced_costs, c_T, x_bar, A_bar, B):
    # make tableau
    tableau = []
    for i in range(z.shape[0]):  # for each row
        row = [z[i][0]]
        if i == 0:
            for c in reduced_costs:
                row.append(c[0])
            row.append(-(c_T @ x_bar)[0][0])
        else:
            row.extend(list(A_bar[i - 1, :]))
            row.extend(x_bar[B[i - 1]])
        tableau.append(row)

    tableau = np.atleast_2d(np.array(tableau))
    return tableau


def is_tableau_optimal(tableau):
    """
    Since we know the tableau is dual useable, if the tableau is primal useable
    by weak duality the tableau must be optimal
    """
    x_bar = tableau[:, -1][1:]
    return primal_usable(x_bar)


def get_tableau_optimization(tableau):
    x_bar = tableau[:, -1][1:]
    cost_to_optimize = np.min(x_bar)

    k_star = np.where(x_bar == cost_to_optimize)[0][0]
    A_bar_k_star = (tableau[k_star + 1, :])[
        1:-1
    ]  # row corresponding to lowest primal sol

    reduced_costs = tableau[0][1:-1]

    optimal_theta = []  # array that returns optimal theta.
    for idx, a in enumerate(A_bar_k_star):
        if a >= 0 or np.isclose(a, 0, rtol=0.000001):
            continue
        if not optimal_theta or (optimal_theta[0] > -reduced_costs[idx] / a):
            optimal_theta = ((-reduced_costs[idx] / a), idx)
    if not optimal_theta:  # if there is no thetas the problem is unbounded
        return False
    # index i forhold til ordning der forlader basis
    leaving_variable_idx = k_star
    entering_variable = optimal_theta[1]  # nummer på variabel der indtræder
    return (leaving_variable_idx, entering_variable)


def get_new_basis(basis_var, leaving_idx, entering_var):
    basis_var[leaving_idx] = entering_var
    return basis_var


def change_tableau_pivot(tableau, new_pivot_col, old_pivot_idx):
    if (tableau[old_pivot_idx + 1, new_pivot_col + 1]) == 0:
        return np.array([[False]])  # ERROR cannot pivot on 0

    tableau[old_pivot_idx + 1, :] /= tableau[
        old_pivot_idx + 1, new_pivot_col + 1
    ]  # normalise pivot idx to equal 1

    pivot_row = tableau[old_pivot_idx + 1, :]
    for idx, element in enumerate(
        tableau[:, new_pivot_col + 1]
    ):  # for every element in the same col as the pivot
        if idx == old_pivot_idx + 1 or element == 0:
            continue
        tableau[idx, :] += -element * pivot_row

    return tableau


def iteration(tableau, basis_var):
    optimization = get_tableau_optimization(tableau)
    old_basis = basis_var.copy()
    old_tableau = tableau.copy()
    if not optimization:
        return "Unbounded"
    leaving_variable_idx = optimization[0]
    entering_variable = optimization[1]
    new_basis = get_new_basis(basis_var, leaving_variable_idx, entering_variable)

    updated_tableau = change_tableau_pivot(
        tableau, entering_variable, leaving_variable_idx
    )
    if (updated_tableau[0, 0]) is False:
        return "cannot pivot"
    return (updated_tableau, new_basis)


def dual_simplex_tableau(A, b, c_T, basis_var):
    """
    Function to compute the optimum of a linear optimization problem using the dual simplex method
    Returns a optimal tableau.
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
    3. c_T: 1xn list
            * A list of the cost coefficients of the cost function.
    4. basis_var: 1xm list [Remeber that the index starts at 0]
            * A list of the starting basis variables.
    #### Example usage
    A = [[1,2,3],
        [3,6,9]]

    b= [1,2]

    c= [1,1,0]

    basis_var = [0,1] #corresponds to x_1,x_2

    optimal_tableau = dual_simplex_tableau(A,b,c,basis_var)
    """
    np.set_printoptions(
        formatter={"all": lambda x: str(fractions.Fraction(x).limit_denominator())}
    )
    A = np.array(A)
    b = np.atleast_2d(np.array(b)).T
    c_T = np.atleast_2d(np.array(c_T))
    B = basis_var
    J = [j for j in range(A.shape[1]) if j not in B]

    A_B = get_A_basis_matrix(A, B)
    c_B = get_c_basis_matrix(c_T, B)
    A_bar = calculate_A_bar_matrix(A, A_B)
    x_bar = get_primal_solution(A, b, B, J)

    c_bar = calculate_reduced_costs(A, A_bar, c_B, c_T, J)
    if not dual_useable(c_bar):  # check dual useable
        return print(
            "There was an error with the provided values, which resulted in a solution, that is not dual useable.\nThis means that the dual simplex method is not applicable."
        )

    Z = get_z_vector(A_bar)
    tableau = create_tableau(Z, c_bar, c_T, x_bar, A_bar, B)

    while is_tableau_optimal(tableau) is False:
        iteration_data = iteration(tableau, basis_var)
        tableau = iteration_data[0]
        basis_var = iteration_data[1]

    return tableau
