from typing import List, Dict, Tuple, Union
import sympy as sp
import cvxpy as cp
import pandas as pd
import numpy as np
from .operators import negation
from .misc import process_neg, is_symbol, timer
# from .predicate import Predicate_dual
# from .preprocess_fol import FOLConverter
# from .objective_function import ObjectiveFunction


class Setup:
    """
    cvxpy.Problem に渡す objective function と constraints
    の生成

    インスタンス化の際に以下の 2 つを引数として渡す
    
    data_dir_path = "./inputs/toy_data"

    file_names_dict = {
        'supervised': ['L1', 'L2', 'L3'],
        'unsupervised': ['U'],
        'rule': ['rules']
    }
    """

    def __init__(self, input_info: dict, objective_function: object) -> None:

        self.problem_info = input_info 
        self.problem_info['KB']           = None
        self.problem_info['lambda_jl']    = None
        self.problem_info['lambda_hi']    = None
        self.problem_info['eta_js']       = None 
        self.problem_info['eta_hat_js']   = None 
        self.problem_info['M']            = None 
        self.problem_info['q']            = None 
        self.problem_info['target_p_idx'] = None 

        self.objective_function = objective_function

        # # degrees of satisfaction
        # self.c1 = input_dict['c1']
        # self.c2 = input_dict['c2']

        # # データ
        # self.L = input_dict['L']
        # self.U = input_dict['U']
        # self.S = input_dict['S']

        # # KB
        # self.KB_origin = input_dict['KB_origin']
        # self.KB = None

        # # ループ用
        # self.len_j = input_dict['len_j']
        # self.len_l = input_dict['len_l']
        # self.len_u = input_dict['len_u']
        # self.len_s = input_dict['len_s']
        # self.len_h = input_dict['len_h']
        # self.len_i = input_dict['len_i']

        # # cvxpy.Variable
        # self.lambda_jl = None
        # self.lambda_hi = None
        # self.eta_js = None
        # self.eta_hat_js = None

        # # coefficients of affine functions
        # self.M = None 
        # self.q = None

        # # obj func
        # self.objective_function = objective_function

        # # predicate
        # self.predicates_dict_tmp = None
        # self.predicates_dict = None
        # self.target_p_name = input_dict['KB_origin']
        # self.target_p_idx  = None

    @timer
    def load_rules(self):
        fol_processor = FOLProcessor(self.problem_info)
        fol_processor()

        self.problem_info['KB'] = fol_processor.KB
        self.problem_info['M']  = fol_processor.M
        self.problem_info['q']  = fol_processor.q
        self.problem_info['predicates_dict'] = fol_processor.predicates_dict

        predicate_names = list(fol_processor.predicates_dict.keys())
        self.problem_info['target_p_idx'] = predicate_names.index(self.problem_info['target_predicate'])
        
    @timer
    def define_cvxpy_variables(self):
        self.problem_info['lambda_jl']  = cp.Variable(shape=(self.problem_info['len_j'], self.problem_info['len_l']), nonneg=True)
        self.problem_info['lambda_hi']  = cp.Variable(shape=(self.problem_info['len_h'], self.problem_info['len_i']), nonneg=True)
        self.problem_info['eta_js']     = cp.Variable(shape=(self.problem_info['len_j'], self.problem_info['len_s']), nonneg=True) 
        self.problem_info['eta_hat_js'] = cp.Variable(shape=(self.problem_info['len_j'], self.problem_info['len_s']), nonneg=True) 
    
    @timer
    def construct_constraints(self):
        len_l = self.problem_info['len_l']
        len_u = self.problem_info['len_u']
        len_s = self.problem_info['len_s']
        len_h = self.problem_info['len_h']
        len_i = self.problem_info['len_i']

        j = self.problem_info['target_p_idx']
        p_name = self.problem_info['target_predicate']

        lambda_jl  = self.problem_info['lambda_jl']
        lambda_hi  = self.problem_info['lambda_hi']
        eta_js     = self.problem_info['eta_js']
        eta_hat_js = self.problem_info['eta_hat_js']

        L = self.problem_info['L'][p_name].values

        c1 = self.problem_info['c1']
        c2 = self.problem_info['c2']

        M = self.problem_info['M']
        start_col = j * len_u
        end_col = start_col + len_u
        M_j = [M_h[:, start_col:end_col] for M_h in M]

        constraints = []

        constraint = 0

        for h in range(len_h):
            for i in range(len_i):
                lmbda = lambda_hi[h, i]
                m_sum = M_j[h][i, :].sum()
                constraint += lmbda * m_sum

        for l in range(len_l):
            lmbda = lambda_jl[j, l]
            y = L[l, -1]
            constraint += -2 * lmbda * y

        for s in range(len_s):
            eta = eta_js[j, s]
            eta_hat = eta_hat_js[j, s]
            constraint += -1 * (eta - eta_hat)
            
        constraints += [
            constraint == 0
        ]

        for l in range(len_l):
            lmbda = lambda_jl[j, l]
            constraints += [
                lmbda <= c1
            ]
        
        for h in range(len_h):
            for i in range(len_i):
                lmbda = lambda_hi[h, i]
                constraints += [
                    lmbda <= c2
                ]

        return constraints
    
    def main(self):
        self.load_rules()
        self.define_cvxpy_variables()

        objective_function = self.objective_function(self.problem_info).construct()
        constraints = self.construct_constraints()
        return objective_function, constraints
    


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




from typing import List, Dict, Tuple, Union
import sympy as sp
import pandas as pd
import numpy as np
from .operators import negation
from .misc import process_neg, is_symbol

class FOLProcessor:

    def __init__(self, problem_info: dict) -> None:

        self.len_j = problem_info['len_j']
        self.len_u = problem_info['len_u']

        self.KB_origin = problem_info['KB_origin']

        self.KB        = None
        self.M         = None
        self.q         = None

        self.predicates_tmp = list(problem_info['L'].keys())
        self.predicates_dict = None

    def _identify_predicates(self, KB: List[List[str]]) -> Dict[str, sp.Symbol]:
        """
        KB 内の述語を特定し，
        各述語の係数を取り出すために
        sympy.Symbol で表現する
        """
        predicates = []

        for formula in KB:
            for item in formula:
                # if item not in ['¬', '∧', '∨', '⊗', '⊕', '→'] and item not in predicates:
                if not is_symbol(item) and item not in predicates:
                    predicates.append(item)

        if set(predicates) != set(self.predicates_tmp):
            raise ValueError("ルールセットが必要十分な述語の種類を含んでいません。")
        
        predicates_dict = {predicate: sp.Symbol(predicate) for predicate in predicates}
        return predicates_dict

    # def _check_implication(self, formula: List[Union[str, sp.Symbol]]):
    def _check_implication(self, formula: List[str]) -> Tuple[bool, Union[None, int]]:
        """
        formula (リスト) について，
        含意記号 '→' の数を調べ，
        その数が 1 以下になっているかを確認する
        """
        
        # 実質，ここには 1 つのインデックスしか入らない
        implication_idxs = []

        for i, item in enumerate(formula):
            if item == '→':
                implication_idxs.append(i)
        
        implication_num = len(implication_idxs)
        
        if implication_num == 0:
            return False, None
        elif implication_num == 1:
            implication_idx = implication_idxs[0]
            return True, implication_idx
        else:
            print('this formula may be invalid')

    def _eliminate_implication(self, formula: List[str]) -> List[str]:
        """
        formula (リスト) 内に含意記号 '→' あれば変換し，消去する 
        """
        implication_flag, target_idx = self._check_implication(formula)

        if implication_flag:
            # 含意記号 '→' を境に formula (list) を 2 つに分ける
            x = formula[:target_idx]
            y = formula[target_idx + 1:]

            # x → y = ¬ x ⊕ y
            x_new = negation(x)
            y_new = y
            new_operator = ['⊕']

            new_formula = x_new + new_operator + y_new
        else:
            new_formula = formula

        return new_formula
       
    def _get_neg_idx_list(self, formula: List[str]) -> Tuple[List[str], List[str]]:
        """
        formula (リスト) 内の '¬' のインデックスリストと
        '¬' 以外のインデックスリストを返す
        """
        neg_idxs = []
        not_neg_idxs = []

        for i, item in enumerate(formula):
            if item == '¬':
                neg_idxs.append(i)
            else:
                not_neg_idxs.append(i)
        
        return neg_idxs, not_neg_idxs

    def _split_neg_idx_list(self, idx_list) -> List[List[int]]:
        """
        インデックスリストを連続する部分リストに分割する
        """
        result = []
        tmp = []

        for i in range(len(idx_list)):
            if not tmp or idx_list[i] == tmp[-1] + 1:
                tmp.append(idx_list[i])
            else:
                result.append(tmp)
                tmp = [idx_list[i]]

        if tmp:
            result.append(tmp)

        return result
    
    def _eliminate_multi_negations(self, formula: List[str]) -> List[str]:
        """
        formula (リスト) 内の連続する '¬' を削除する
        """
        neg_idxs, not_neg_idxs = self._get_neg_idx_list(formula)
        neg_idxs_decomposed = self._split_neg_idx_list(neg_idxs)

        neg_idxs_new = []
        for tmp in neg_idxs_decomposed:
            if len(tmp) % 2 == 0:
                pass
            else:
                neg_idxs_new.append(tmp[0])
        
        idxs_new = sorted(neg_idxs_new + not_neg_idxs)
        
        formula_new = []
        for idx in idxs_new:
            item = formula[idx]
            formula_new.append(item)

        return formula_new
    
    def convert_KB_origin(self) -> None:
        self.KB = []
        for formula in self.KB_origin:
            new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(new_formula)
            self.KB.append(new_formula)
            
    def calculate_M_and_q(self) -> None:
        self.predicates_dict = self._identify_predicates(self.KB)

        # sympy で predicate を構成した KB
        # （formula を式変形した後の predicate の係数を取り出すため）
        KB_tmp = []
        for formula in self.KB:

            tmp_formula = []
            for item in formula:
                if item in self.predicates_dict.keys():
                    tmp_formula.append(self.predicates_dict[item])
                else:
                    tmp_formula.append(item)

            tmp_formula = self._eliminate_multi_negations(tmp_formula)

            process_neg(tmp_formula, is_1_symbol=True)

            phi_h = []
            new_formula_1 = [sp.Symbol('1')]
            new_formula_2 = []

            tmp_new_formula_2 = 0
            for item in tmp_formula:

                # この処理間違ってないか -> 合っている （symbol ex. ⊕）
                if not is_symbol(item):
                    tmp_new_formula_2 += item
            
            new_formula_2.append(tmp_new_formula_2)

            phi_h.append(new_formula_1)
            phi_h.append(new_formula_2)
        
            KB_tmp.append(phi_h)

        predicates = list(self.predicates_dict.values())

        self.M = []
        self.q = []

        for phi_h in KB_tmp:

            base_M_h = np.zeros((len(phi_h), self.len_j))
            base_q_h = np.zeros((len(phi_h), 1))
            for i, formula in enumerate(phi_h):
                for j, predicate in enumerate(predicates):

                    # negation
                    val = sp.Symbol('1') - formula[0]
                    coefficient = val.coeff(predicate)
                    base_M_h[i, j] = coefficient
                
                val = sp.Symbol('1') - formula[0]
                base_q_h[i] = val.coeff(sp.Symbol('1'))

            tmp_M_h = []
            for i in range(self.len_j):
                column = base_M_h[:, i]
                zeros = np.zeros((len(phi_h), self.len_u - 1))
                concatenated_column = np.concatenate((column[:, np.newaxis], zeros), axis=1)
                tmp_M_h.append(concatenated_column)

            tmp_M_h = [np.concatenate(tmp_M_h, axis=1)]
            
            shifted_M_h = tmp_M_h[0]

            for i in range(self.len_u - 1):
                shifted_M_h = np.roll(shifted_M_h, 1, axis=1)
                tmp_M_h.append(shifted_M_h)
            
            M_h = np.concatenate(tmp_M_h, axis=0)
            self.M.append(M_h)

            
            tmp_q_h = [base_q_h for _ in range(self.len_u)]
            q_h = np.concatenate(tmp_q_h, axis=0)
            self.q.append(q_h)

    def __call__(self) -> Tuple[Union[None, pd.DataFrame], List[List[str]], List[List[str]], List[np.ndarray], List[np.ndarray], Dict[str, sp.Symbol]]:
        self.convert_KB_origin()
        self.calculate_M_and_q()



import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score
import optuna

from src.misc import linear_kernel

class Predicate_dual:
    def __init__(
            self, 
            problem_info: dict,
            metrics: sklearn.metrics = None,
            opt_iter_num: int = 100, 
            opt_range_size: float = 5
        ) -> None:
    
        self.p_name = problem_info['target_predicate']
        self.j      = problem_info['target_p_idx']

        self.c1 = problem_info['c1']
        self.c2 = problem_info['c2']

        self.L = problem_info['L'][self.p_name].values
        self.U = problem_info['U'][self.p_name]
        self.S = problem_info['S'][self.p_name]

        self.len_l = problem_info['len_l']
        self.len_u = problem_info['len_u']
        self.len_s = problem_info['len_s']
        self.len_h = problem_info['len_h'] 
        self.len_i = problem_info['len_i']

        self.lambda_jl  = problem_info['lambda_jl'][self.j, :].value
        self.lambda_hi  = problem_info['lambda_hi'].value
        self.eta_js     = problem_info['eta_js'][self.j, :].value
        self.eta_hat_js = problem_info['eta_hat_js'][self.j, :].value

        M = problem_info['M']
        start_col = self.j * self.len_u
        end_col = start_col + self.len_u
        self.M_j = [M_h[:, start_col:end_col] for M_h in M]

        self.k = problem_info['kernel_function']

        if metrics == None:
            self.metrics = accuracy_score
        else:
            self.metrics = metrics

        self.n_trials = opt_iter_num
        self.range_size = opt_range_size
        self.b = self.optimize_b()

        self.w_linear_kernel = self._w_linear_kernel() 
        self.coeff = np.append(self.w_linear_kernel, self.b)
    
    def compute_kernel_matrix(self, X1, X2) -> np.ndarray:
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
    
    def w_dot_phi(self, x_pred: np.ndarray) -> float:
        values = np.zeros(len(x_pred))
        
        x = self.L[:, :-1]
        y = self.L[:, -1]
        lmbda = self.lambda_jl
        K = self.compute_kernel_matrix(x, x_pred)

        values += 2 * (y * lmbda).T @ K 

        x = self.U
        lmbda = self.lambda_hi.flatten().reshape(-1, 1)
        M = np.vstack(self.M_j)
        K = self.compute_kernel_matrix(x, x_pred)

        values += (-1) * (lmbda * M @ K).sum(axis=0)

        x = self.S
        eta = self.eta_js
        eta_hat = self.eta_hat_js
        K = self.compute_kernel_matrix(x, x_pred)

        values += (eta - eta_hat) @ K
        
        return values
     
    def _w_linear_kernel(self) -> np.ndarray:
        """
        This should be rewritten using matrix calculations.
        """

        input_dim = len(self.L[0, :-1])
        w_linear_kernel = np.zeros(input_dim)

        for l in range(self.len_l):
            x = self.L[l, :-1]
            y = self.L[l, -1]
            lmbda = self.lambda_jl[l]
            w_linear_kernel += 2 * lmbda * y * x

        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                for u in range(self.len_u):
                    x = self.U[u]
                    M = self.M_j[h][i, u]
                    w_linear_kernel += - lmbda * M * x
        
        for s in range(self.len_s):
            x = self.S[s]
            eta = self.eta_js[s]
            eta_hat = self.eta_hat_js[s]
            w_linear_kernel += (eta - eta_hat) * x

        return w_linear_kernel
    
    def calculate_initial_b(self) -> float:
        x = self.L[:, :-1]
        y = self.L[:, -1]

        initial_b = np.mean(y - self.w_dot_phi(x))

        return initial_b
    
    def calculate_score_at_b(
            self, 
            b: float,
            X: np.ndarray,
            y: np.ndarray
        ) -> float:

        y_pred = self.w_dot_phi(X)

        y_pred = np.array(y_pred) + b
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

        score = self.metrics(y, y_pred_interpreted)

        return score
    
    def optimize_b(self):
        initial_b = self.calculate_initial_b()

        X_train = self.L[:, :-1]
        y_train = self.L[:, -1]

        range_size = self.range_size

        min_bound = initial_b - 0.5 * range_size
        max_bound = initial_b + 0.5 * range_size
        
        print()
        print(f'min_bound: {min_bound}')
        print(f'max_bound: {max_bound}')
        print()

        def objective(trial):
            b = trial.suggest_float('b', min_bound, max_bound)
            return self.calculate_score_at_b(b, X_train, y_train)  
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        study.enqueue_trial(initial_b)

        optimal_b = study.best_params['b']
        return optimal_b

    def __call__(self, x_pred: np.ndarray) -> float:
        
        value = self.w_dot_phi(x_pred) + self.b

        return value