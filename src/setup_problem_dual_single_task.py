import cvxpy as cp
from .misc import timer
from .preprocess_fol import FOLProcessor



class Setup:
    """
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
    

