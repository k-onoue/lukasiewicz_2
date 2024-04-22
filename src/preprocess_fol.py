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

        # if set(predicates) != set(self.predicates_tmp):
        #     raise ValueError("ルールセットが必要十分な述語の種類を含んでいません。")
        
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

    def __call__(self) -> None:
        self.convert_KB_origin()
        self.calculate_M_and_q()
