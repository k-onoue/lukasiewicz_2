import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import optuna


class Predicate_dual:
    def __init__(
            self, 
            problem_info: dict,
            metrics: str = None,
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

        self.metrics_dict = {
            'accuracy': accuracy_score,
            'f1'      : f1_score
        }

        if metrics == None:
            self.metrics = "f1"
        else:
            if metrics not in self.metrics_dict.keys():
                ValueError("invalid metrics to optimize b")
            else: self.metrics = metrics

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
        - X2: Second input matrix (l x m)
        - kernel_function: Kernel function to use (default is dot product)

        Returns:
        - Kernel matrix (n x l)
        """
        kernel_function = self.k

        K_matrix = kernel_function(X1, X2)
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

        metrics = self.metrics_dict[self.metrics]
        score = metrics(y, y_pred_interpreted)

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