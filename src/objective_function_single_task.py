from typing import List, Tuple
import cvxpy as cp
import numpy as np
from .misc import timer
from sklearn.metrics.pairwise import linear_kernel


class ObjectiveFunction:
    """
    A class to construct the objective function for the optimization problem.

    Attributes
    ----------
    problem_info : dict
        Dictionary containing problem-settings.
    etc. 

    Methods
    -------
    compute_kernel_matrix(X1, X2):
        Compute the kernel matrix between two matrices.
    _mapping_variables():
        Guide to construct P.
    _construct_P_j(mapping_x_i, x):
        Constructs the matrix P for the quadratic form in the objective function.
    construct():
        Constructs the objective function for the optimization problem.
    """
    
    def __init__(self, problem_info: dict, kernel_function: object = linear_kernel) -> None:
        """
        Initializes the ObjectiveFunction class with problem-specific information and a kernel function.

        Parameters
        ----------
        problem_info : dict
            Dictionary containing problem-specific information and parameters.
        kernel_function : callable, optional
            The kernel function to use (default is linear_kernel).
        """
        self.p_name = problem_info['target_predicate']
        self.j = problem_info['target_p_idx']

        self.L = problem_info['L'][self.p_name].values
        self.U = problem_info['U'][self.p_name]
        self.S = problem_info['S'][self.p_name]

        self.len_l = problem_info['len_l']
        self.len_u = problem_info['len_u']
        self.len_s = problem_info['len_s']
        self.len_h = problem_info['len_h']
        self.len_i = problem_info['len_i']

        self.lambda_jl = problem_info['lambda_jl'][self.j]
        self.lambda_hi = problem_info['lambda_hi']
        self.eta_js = problem_info['eta_js'][self.j]
        self.eta_hat_js = problem_info['eta_hat_js'][self.j]

        self.M = problem_info['M']
        self.q = problem_info['q']

        self.k = problem_info['kernel_function']
    
    def compute_kernel_matrix(self, X1, X2):
        """
        Compute the kernel matrix between two matrices.

        Parameters
        ----------
        X1 : np.ndarray
            First input matrix (n x m).
        X2 : np.ndarray
            Second input matrix (l x m).

        Returns
        -------
        np.ndarray
            Kernel matrix (n x l).
        """
        kernel_function = self.k

        K_matrix = kernel_function(X1, X2)
        return K_matrix
     
    def _mapping_variables(self) -> Tuple[dict, List[cp.Variable]]:
        """
        Maps the variables to indices and creates a list of variables.

        Returns
        -------
        tuple
            A tuple containing a dictionary mapping variable names to indices and a list of CVXPY variables.
        """
        mapping_x_i = {}
        x = []

        mapping_x_i["lambda_jl"] = {}
        for l in range(self.len_l):
            mapping_x_i['lambda_jl'][(0, l)] = len(x)
            x.append(self.lambda_jl[l])

        mapping_x_i["lambda_hi"] = {}
        for h in range(self.len_h):
            for i in range(self.len_i):
                mapping_x_i["lambda_hi"][(h, i)] = len(x)
                x.append(self.lambda_hi[h, i])

        mapping_x_i['delta_eta_js'] = {}
        for s in range(self.len_s):
            mapping_x_i["delta_eta_js"][(0, s)] = len(x)
            x.append(self.eta_js[s] - self.eta_hat_js[s])

        return mapping_x_i, x
    
    @timer
    def _construct_P_j(self, mapping_x_i: dict, x: List[cp.Variable]) -> Tuple[cp.Variable, np.ndarray]:
        """
        Constructs the matrix P for the quadratic form in the objective function.

        Parameters
        ----------
        mapping_x_i : dict
            A dictionary mapping variable names to indices.
        x : list of cp.Variable
            A list of CVXPY variables.

        Returns
        -------
        tuple
            A tuple containing the CVXPY variable and the matrix P.
        """
        P = np.zeros((len(x), len(x)))
        print(f'shape of P: {P.shape}')

        start_col = self.j * self.len_u
        end_col = start_col + self.len_u
        M = [M_h[:, start_col:end_col] for M_h in self.M]

        # P_{11} --------------------------------------------------
        x_L = self.L[:, :-1]
        K = self.compute_kernel_matrix(x_L, x_L)
        
        y_L = self.L[:, -1].reshape(-1, 1)
        Y = y_L @ y_L.T 

        P_11 = 4 * Y * K

        start = mapping_x_i['lambda_jl'][(0, 0)]
        end = mapping_x_i['lambda_jl'][(0, self.len_l - 1)] + 1

        P[start:end, start:end] = P_11
        print('finish l')

        # P_{22} --------------------------------------------------
        K = self.compute_kernel_matrix(self.U, self.U)
        M_vstacked = np.vstack(M)

        P_22 = M_vstacked @ K @ M_vstacked.T

        start = mapping_x_i['lambda_hi'][(0, 0)]
        end = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1

        P[start:end, start:end] = P_22
        print('finish h')

        # P_{33}--------------------------------------------------
        K = self.compute_kernel_matrix(self.S, self.S)
        
        P_33 = K

        start = mapping_x_i['delta_eta_js'][(0, 0)]
        end = mapping_x_i['delta_eta_js'][(0, self.len_s - 1)] + 1

        P[start:end, start:end] = P_33
        print('finish s')

        # P_{12} --------------------------------------------------
        K = self.compute_kernel_matrix(self.L[:, :-1], self.U)
        M_vstacked = np.vstack(M)
        y_L = self.L[:, -1].reshape(-1, 1)

        P_12 = (-4) * y_L * K @ M_vstacked.T

        r_start = mapping_x_i['lambda_jl'][(0, 0)]
        r_end = mapping_x_i['lambda_jl'][(0, self.len_l - 1)] + 1
        c_start = mapping_x_i['lambda_hi'][(0, 0)]
        c_end = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1

        P[r_start:r_end, c_start:c_end] = P_12
        print('finish l h')

        # P_{13} --------------------------------------------------
        x_L = self.L[:, :-1]
        K = self.compute_kernel_matrix(x_L, self.S)

        y_L = self.L[:, -1].reshape(-1, 1)
        P_13 = 4 * y_L * K

        r_start = mapping_x_i['lambda_jl'][(0, 0)]
        r_end = mapping_x_i['lambda_jl'][(0, self.len_l - 1)] + 1
        c_start = mapping_x_i['delta_eta_js'][(0, 0)]
        c_end = mapping_x_i['delta_eta_js'][(0, self.len_s - 1)] + 1

        P[r_start:r_end, c_start:c_end] = P_13
        print('finish l s')

        # P_{23} --------------------------------------------------
        K = self.compute_kernel_matrix(self.U, self.S)
        M_vstacked = np.vstack(M)
        P_23 = M_vstacked @ K
        
        r_start = mapping_x_i['lambda_hi'][(0, 0)]
        r_end = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1
        c_start = mapping_x_i['delta_eta_js'][(0, 0)]
        c_end = mapping_x_i['delta_eta_js'][(0, self.len_s - 1)] + 1

        P[r_start:r_end, c_start:c_end] = (-2) * P_23
        print('finish h s')

        P = (P + P.T) / 2
        return cp.vstack(x), P

    def construct(self) -> cp.Expression:
        """
        Constructs the objective function for the optimization problem.

        Returns
        -------
        cp.Expression
            The constructed objective function.
        """
        objective_function = 0

        mapping_x_i, x = self._mapping_variables()
        x, P = self._construct_P_j(mapping_x_i, x)

        P += np.diag(np.ones(P.shape[0])) * 1e-6

        objective_function = (-1 / 2) * cp.quad_form(x, cp.psd_wrap(P))

        for l in range(self.len_l):
            objective_function += self.lambda_jl[l]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                objective_function += self.lambda_hi[h, i] * (1 / 2 * self.M[h][i, :].sum() + self.q[h][i, 0])

        for s in range(self.len_s):
            objective_function += (-1 / 2) * (self.eta_js[s] + self.eta_hat_js[s])

        objective_function = cp.Maximize(objective_function)

        return objective_function
