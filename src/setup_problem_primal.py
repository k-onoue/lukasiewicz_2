from typing import List, Tuple, Union
import cvxpy as cp
import numpy as np
from src.operators import negation
from src.misc import timer, is_symbol, process_neg


# primal の predicate
class PredicatePrimal:
    def __init__(self, w: cp.Variable) -> None:
        self.w = w

    def __call__(self, x: np.ndarray) -> cp.Expression:
        w = self.w[:-1]
        b = self.w[-1]
        return w @ x.T + b


def log_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> cp.Expression:
    """
    log_loss，クロスエントロピー
    scikit-learn の log_loss だと，
    途中必ず実数値で計算しないといけない場所（np.clip）が出てきて
    cvxpy の中で使用するとエラーが出たので実装
    """
    y_true = np.where(y_true == -1, 0, y_true)
    losses = - (y_true @ cp.log(y_pred) + (1 - y_true) @ cp.log(1 - y_pred))
    average_loss = np.mean(losses)
    return average_loss


class FOLProcessorPrimal:

    def __init__(self, problem_info: dict) -> None:

        self.KB_origin       = problem_info['KB_origin']
        self.KB              = None
        self.predicates_tmp = list(problem_info['L'].keys())
        self.predicates_dict = None
    
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
        """
        新しい KB を返す
        """
        self.KB = []

        for formula in self.KB_origin:
            new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(new_formula)
            self.KB.append(new_formula)
        
    def identify_predicates(self) -> None:
        predicates = []

        for formula in self.KB:
            for item in formula:
                # if item not in ['¬', '∧', '∨', '⊗', '⊕', '→'] and item not in predicates:
                if not is_symbol(item) and item not in predicates:
                    predicates.append(item)

        if set(predicates) != set(self.predicates_tmp):
            raise ValueError("ルールセットが必要十分な述語の種類を含んでいません。")
        
        self.predicates_dict = {predicate: 0 for predicate in predicates}

    def __call__(self) -> None:
        self.convert_KB_origin()
        self.identify_predicates()




class SetupPrimal:
    def __init__(
            self, 
            input_info:dict
        ) -> None:

        self.problem_info                 = input_info
        self.problem_info['KB']           = None
        self.problem_info['w_j']          = None
        self.problem_info['xi_jl']        = None
        self.problem_info['xi_h']         = None
        self.problem_info['target_p_idx'] = None

    @timer
    def load_rules(self):
        """
        .txt ファイルとして保存されている Knowledge Base (KB) を読み込み，
        リストとして保持する
        """
        fol_processor = FOLProcessorPrimal(self.problem_info)
        fol_processor()

        self.problem_info['KB'] = fol_processor.KB

        self._define_cvxpy_variables()

        w_j = self.problem_info['w_j']

        predicates_dict = {}
        for j, predicate in enumerate(fol_processor.predicates_dict.keys()):
            predicates_dict[predicate] = PredicatePrimal(w_j[j])

        self.problem_info['predicates_dict'] = predicates_dict

    @timer
    def _define_cvxpy_variables(self) -> None:
        len_j   = self.problem_info['len_j']
        len_l   = self.problem_info['len_l']
        len_h   = self.problem_info['len_h']

        L       = self.problem_info['L']
        L_tmp   = next(iter(L.values())).values
        dim_x_L = len(L_tmp[0, :-1]) + 1

        self.problem_info['w_j']   = cp.Variable(shape=(len_j, dim_x_L))
        self.problem_info['xi_jl'] = cp.Variable(shape=(len_j, len_l), nonneg=True)
        self.problem_info['xi_h']  = cp.Variable(shape=(len_h, 1), nonneg=True)

    def _calc_KB_at_datum(self, KB, datum):
        """
        logical constraints を構成する際に使用．
        KB の中のすべての predicate をあるデータ点で計算する
        """
        predicates_dict = self.problem_info['predicates_dict']

        KB_new = []

        for formula in KB:
            new_formula = []

            for item in formula:
                if item in predicates_dict:
                    new_formula.append(predicates_dict[item](datum))
                else:
                    new_formula.append(item)

            process_neg(new_formula)
            KB_new.append(new_formula)

        return KB_new

    @timer
    def _construct_consistency_constraints(self) -> List[cp.Expression]:
        """
        consistency constraints を構成する
        """
        predicates_dict = self.problem_info['predicates_dict']
        S               = self.problem_info['S']

        constraints_tmp = []
        for p_name, p in predicates_dict.items():
            for x_s in S[p_name]:
                constraints_tmp += [
                    p(x_s) <= 1,
                    p(x_s) >= 0
                ]

        print("consistency constraints")
        return constraints_tmp
    
    # logistic regression
    @timer
    def logistic_regression_loss(self) -> cp.Expression:
        """
        目的関数を構成する．
        c1 は logical constraints を，
        c2 は consistency constraints を
        満足する度合いを表す．
        """
        KB = self.problem_info['KB']

        len_j = self.problem_info['len_j']
        w_j   = self.problem_info['w_j']

        predicates_dict = self.problem_info['predicates_dict']
        c1  = self.problem_info['c1']
        c2  = self.problem_info['c2']
        L   = self.problem_info['L']
        U   = next(iter(self.problem_info['U'].values()))
        w_j = self.problem_info['w_j']

        print(f'obj coeff')
        print(f'c1: {c1}')
        print(f'c2: {c2}')

        function = 0

        for j in range(len_j):
            w = w_j[j, :-1]
            function += 1/2 * (cp.norm2(w) ** 2)

        for p_name, p in predicates_dict.items():
            x = L[p_name].values[:, :-1]
            y = L[p_name].values[:, -1]
            y_pred = p(x)
            value = log_loss(y, y_pred)
            function += c1 * value 

        for u in U:
            KB_tmp = self._calc_KB_at_datum(KB, u)
            for formula in KB_tmp:
                formula_tmp = 0
                for item in formula:
                    if not is_symbol(item):
                        formula_tmp += item
                tmp = cp.maximum(0, negation(formula_tmp))
                function += c2 * tmp

        objective_function = cp.Minimize(function)
        return objective_function

    def main(self):
        """
        目的関数と制約の構成．
        """
        self.load_rules()

        objective_function = self.logistic_regression_loss()
        constraints = self._construct_consistency_constraints()

        return objective_function, constraints
    


