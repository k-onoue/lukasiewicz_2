import time
from typing import List, Union

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

from .operators import negation

symbols = ['¬', '∧', '∨', '⊗', '⊕', '→'] + ['∀', '∃']



# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    def __init__(self):
        pass


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time} seconds!')
        return result
    return wrapper 
    

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> cp.Expression:
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
    

def _count_neg(formula_decomposed: List[Union[str, cp.Expression]]) -> int:
    """
    process_neg 関数の中で使用．
    cvxpy.Variable と str が混ざると
    リストに対する組み込み関数での操作でエラーが出たため実装
    list.count(x)

    formula (list) 内の '¬' の数を数える


    """
    neg_num = 0
    
    for item in formula_decomposed:
        if isinstance(item, str):
            if item == '¬':
                neg_num += 1

    return neg_num


def _get_first_neg_index(formula_decomposed: List[Union[str, cp.Expression]]) -> int:
    """
    process_neg 関数の中で使用．
    cvxpy.Variable と str が混ざると
    リストに対する組み込み関数での操作でエラーが出たため実装
    list.index(x)

    formula (list) 内の初めの '¬' のインデックスを取得
    """
    target_index = None

    for i, item in enumerate(formula_decomposed):
        if type(item) == str:
            if item == '¬':
                target_index = i
                break
    
    return target_index


def process_neg(formula: List[Union[str, cp.Expression]], is_1_symbol: bool = False) -> None:
    """
    formula（list）に含まれている
    否定記号 '¬' を変換し，消去する 
    """
    neg_num = _count_neg(formula)

    while neg_num > 0:
        target_index = _get_first_neg_index(formula)

        # 演算に使用する値を取得
        x = formula[target_index + 1]

        # 演算の実行
        result = negation(x, is_1_symbol=is_1_symbol)

        # 演算結果で置き換え，演算子（¬）の削除
        formula[target_index + 1] = result
        formula.pop(target_index)

        neg_num -= 1

    # return formula


def count_specific_operator(formula_decomposed: List[Union[str, cp.Expression]], operator: str) -> int:
    """
    formula（list）について，
    特定の演算記号の数を数える
    """
    neg_num = 0
    
    for item in formula_decomposed:
        if type(item) == str:
            if item == operator:
                neg_num += 1

    return neg_num


def get_first_specific_oprator_index(formula_decomposed: List[Union[str, cp.Expression]], operator: str) -> int:
    """
    formula (list) について，
    特定の演算記号のインデックスのうち，
    一番小さいものを取得
    """
    target_index = None

    for i, item in enumerate(formula_decomposed):
        if type(item) == str:
            if item == operator:
                target_index = i
                break
    
    return target_index

def is_symbol(item: Union[str, cp.Expression]) -> bool:
    """
    リストとして保持されている formula の要素が演算記号であるかを判定
    """
    flag = False

    if type(item) != str:
        return flag
    else:
        for symbol in symbols:
            if item == symbol:
                flag = True
        return flag


def boundary_equation_2d(x1: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """
    境界条件の方程式
    入力データの次元が 2 のときのみ使用可能
    """
    w1 = coeff[0]
    w2 = coeff[1]
    b  = coeff[2]

    x = np.hstack([x1, np.ones_like(x1)])
    w = np.array([-w1/w2, -b/w2 + 0.5/w2]).reshape(-1,1)

    return x @ w


def visualize_result(problem_instance: Setup_, colors=['red', 'blue', 'green', 'yellow', 'black']) -> None:
    """
    入力データの次元が 2 のときのみ使用可能
    """
    L = problem_instance.L
    w_j = problem_instance.w_j.value
    len_j = problem_instance.len_j
    len_l = problem_instance.len_l

    test_x = np.linspace(0.05, 0.95, 100).reshape(-1, 1)
    test_ys = []
    for w in w_j:
        test_ys.append(boundary_equation_2d(test_x, w))

    plt.figure(figsize=(6,4))
    
    for j, p_name in enumerate(problem_instance.predicates_dict.keys()):
        for l in range(len_l):
            if L[p_name][l, 2] == 1:
                plt.scatter(L[p_name][l,0], L[p_name][l,1], c=colors[j], marker='o', label='1')
            else:
                plt.scatter(L[p_name][l,0], L[p_name][l,1], facecolors='none', edgecolors=colors[j], marker='o', label='-1')

    for j, test_y in enumerate(test_ys):
        plt.plot(test_x, test_y, label=f'p{j+1}')
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()
