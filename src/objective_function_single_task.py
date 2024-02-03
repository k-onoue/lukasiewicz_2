
from typing import List, Tuple
import cvxpy as cp
import numpy as np
from .misc import timer
from src.misc import linear_kernel


class ObjectiveFunction:
    def __init__(
            self, 
            problem_info: dict, 
            kernel_function: object = linear_kernel
        ) -> None:

        self.p_name = problem_info['target_predicate']
        self.j      = problem_info['target_p_idx']

        self.L = problem_info['L'][self.p_name].values
        self.U = problem_info['U'][self.p_name]
        self.S = problem_info['S'][self.p_name]

        self.len_l = problem_info['len_l']
        self.len_u = problem_info['len_u']
        self.len_s = problem_info['len_s']
        self.len_h = problem_info['len_h'] 
        self.len_i = problem_info['len_i']

        self.lambda_jl  = problem_info['lambda_jl'][self.j]
        self.lambda_hi  = problem_info['lambda_hi']
        self.eta_js     = problem_info['eta_js'][self.j]
        self.eta_hat_js = problem_info['eta_hat_js'][self.j]

        self.M = problem_info['M'] 
        self.q = problem_info['q']

        self.k = problem_info['kernel_function']
    
    def compute_kernel_matrix(self, X1, X2):
        """
        Compute the kernel matrix between two matrices.

        Parameters:
        - X1: First input matrix (n x m)
        - X2: Second input matrix (n x m)
        - kernel_function: Kernel function to use (default is dot product)

        Returns:
        - Kernel matrix (n x n)
        """
        kernel_function = self.k

        K_matrix = kernel_function(X1, X2.T)
        return K_matrix
     
    def _mapping_variables(self) -> Tuple[dict, List[cp.Variable]]:
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
    def _construct_P_j(
        self,
        mapping_x_i: dict,
        x: List[cp.Variable]
    ) -> Tuple[cp.Variable, np.ndarray]: 

        P = np.zeros((len(x), len(x)))

        ############################################
        ############################################
        ############################################
        print(f'shape of P: {P.shape}')
        ############################################
        ############################################
        ############################################

        start_col = self.j * self.len_u
        end_col = start_col + self.len_u

        M = [M_h[:, start_col:end_col] for M_h in self.M]

        # P_{11}
        for l in range(self.len_l):
            for l_ in range(self.len_l):
                row = mapping_x_i['lambda_jl'][(0, l)]
                col = mapping_x_i['lambda_jl'][(0, l_)]

                x_l  = self.L[l, :-1]
                x_l_ = self.L[l_, :-1]
                k    = self.k(x_l, x_l_)

                y_l  = self.L[l, -1]
                y_l_ = self.L[l_, -1]

                P[row, col] += 4 * y_l * y_l_ * k


        ############################################
        ############################################
        print('finish l')
        ############################################
        ############################################


        # P_{22} using matrix multiplication
        K = self.compute_kernel_matrix(self.U, self.U)
        M_vstacked = np.vstack(M)

        P_22 = M_vstacked @ K @ M_vstacked.T

        start = mapping_x_i['lambda_hi'][(0, 0)]
        end   = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1

        P[start:end, start:end] = P_22

        
        ############################################
        ############################################
        print('finish h')
        ############################################
        ############################################

        # P_{33}
        for s in range(self.len_s):
            for s_ in range(self.len_s):
                row = mapping_x_i['delta_eta_js'][(0, s)]
                col = mapping_x_i['delta_eta_js'][(0, s_)]

                x_s  = self.S[s]
                x_s_ = self.S[s_]
                k  = self.k(x_s, x_s_)

                P[row, col] += k

        ############################################
        ############################################
        print('finish s')
        ############################################
        ############################################

        # P_{12}
        K = self.compute_kernel_matrix(self.L[:, :-1], self.U)
        M_vstacked = np.vstack(M)
        y_L = self.L[:, -1].reshape(-1, 1)

        P_12 = (-4) * y_L * K @ M_vstacked.T

        r_start = mapping_x_i['lambda_jl'][(0, 0)]
        r_end   = mapping_x_i['lambda_jl'][(0, self.len_l - 1)] + 1
        c_start = mapping_x_i['lambda_hi'][(0, 0)]
        c_end   = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1

        P[r_start:r_end, c_start:c_end] = P_12

        
        ############################################
        ############################################
        print('finish l h')
        ############################################
        ############################################

        # P_{13}
        for l in range(self.len_l):
            for s in range(self.len_s):
                row = mapping_x_i['lambda_jl'][(0, l)]
                col = mapping_x_i['delta_eta_js'][(0, s)]

                y_l = self.L[l, -1]

                x_l = self.L[l, :-1]
                x_s = self.S[s]
                k = self.k(x_l, x_s)

                P[row, col] += 4 * y_l * k
        

        ############################################
        ############################################
        print('finish l s')
        ############################################
        ############################################

        # P_{23}
        K = self.compute_kernel_matrix(self.U, self.S)
        M_vstacked = np.vstack(M)
        P_23 = M_vstacked @ K
        
        r_start = mapping_x_i['lambda_hi'][(0, 0)]
        r_end   = mapping_x_i['lambda_hi'][(self.len_h - 1, self.len_i - 1)] + 1
        c_start = mapping_x_i['delta_eta_js'][(0, 0)]
        c_end   = mapping_x_i['delta_eta_js'][(0, self.len_s - 1)] + 1

        P[r_start:r_end, c_start:c_end] = (-2) * P_23

        ############################################
        ############################################
        print('finish h s')
        ############################################
        ############################################

        P = (P+P.T)/2
        return cp.vstack(x), P
                 

    def construct(self) -> cp.Expression:

        objective_function = 0

        mapping_x_i, x = self._mapping_variables()
        x, P = self._construct_P_j(mapping_x_i, x)

        # 計算安定性のため
        P += np.diag(np.ones(P.shape[0])) * 1e-6

        objective_function = (-1/2) * cp.quad_form(x, P)

        for l in range(self.len_l):
            objective_function += self.lambda_jl[l]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                objective_function += self.lambda_hi[h, i] * (1/2 * self.M[h][i, :].sum() + self.q[h][i, 0])

        for s in range(self.len_s):
            objective_function += (-1/2) * (self.eta_js[s] + self.eta_hat_js[s])

        objective_function = cp.Maximize(objective_function)

        return objective_function

