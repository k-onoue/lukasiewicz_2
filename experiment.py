import json
import os
from functools import partial

import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import cvxpy as cp

from src.misc import is_symbol
# from src.misc import linear_kernel
# from src.misc import rbf_kernel

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel

from src.setup_problem_dual_single_task import Setup
from src.objective_function_single_task import ObjectiveFunction
from src.predicate_single_task import Predicate_dual

from src.evaluation import evaluate_model



# 入力ファイル
file_path_1 = "data/pima_indian_diabetes/diabetes_cleaned_normalized.csv"
file_path_2 = "data/pima_indian_diabetes/diabetes_discretized.csv"
file_path_3 = "data/pima_indian_diabetes/rules_3.txt"


df_origin_1 = pd.read_csv(file_path_1, index_col=0).reset_index(drop=True)
X_origin_1 = df_origin_1.drop(["Outcome"], axis=1)
y_origin_1 = df_origin_1["Outcome"]

df_origin_2 = pd.read_csv(file_path_2, index_col=0).reset_index(drop=True)
print(df_origin_1.head())
print(df_origin_2.head())


# 実験設定
settings = {
    'path': './experiments',
    'source_paths': [file_path_1, file_path_2, file_path_3],
    'experiment_name': 'pima_indian_diabetes_cv_1',
    'seed': 42,
    'n_splits': 5,
    'n_unsupervised': 15,
    'c1': 10,
    'c2': 10,
    'result': {}
}



kf = KFold(n_splits=settings['n_splits'])

idx_split = {}

for i, (train_idx, test_idx) in enumerate(kf.split(df_origin_1)):

    print()
    print()
    print()
    print()
    print()
    print(f"fold: {i+1} of {settings['n_splits']}")

    idx_split[i] = train_idx.tolist(), test_idx.tolist()

    # 訓練データ（提案モデル用）--------------------------------------------
    L = {}
    for col_name in df_origin_2.columns:
        df_new = X_origin_1.copy().iloc[train_idx, :]
        df_new['target'] = df_origin_2[col_name].replace(0, -1)
        L[col_name] = df_new

    np.random.seed(seed=settings['seed'])
    arr_u = np.random.rand(settings['n_unsupervised'], X_origin_1.shape[1])
    U = {key: arr_u for key in L.keys()}

    S = {key: np.vstack([df.drop(['target'], axis=1).values, arr_u]) for key, df in L.items()}

    # ルール
    KB_origin =  []

    with open(file_path_3, 'r') as file:
        for line in file:
            formula = line.split()
            KB_origin.append(formula)

    # パラメータ
    len_j = len(L)
    len_l = len(train_idx)
    len_u = settings['n_unsupervised']
    len_s = len_l + len_u

    len_h = len(KB_origin)
    len_i = len_u * 2

    # テストデータ -------------------------------------------------------
    df_tmp = df_origin_1.copy().iloc[test_idx, :]
    df_tmp= df_tmp.rename(columns={'Outcome': 'target'})
    df_tmp['target'] = df_tmp['target'].replace(0, -1)
    
    # from src.misc import is_symbol
    rules_tmp = []
    for rule in KB_origin:
        if "Outcome" in rule:
            tmp = {}
            for idx, item in enumerate(rule):
                if not is_symbol(item):
                    if idx == 0 or rule[idx - 1] != '¬':
                        tmp[item] = 1
                    elif item != "Outcome":
                        tmp[item] = 0
                    else:
                        tmp[item] = -1

            rules_tmp.append(tmp)

    rule_violation_check = {}

    for h, rule in enumerate(rules_tmp):
        outcome = rule['Outcome']

        condition_parts = [
            f"{column} == {value}" 
            for column, value in rule.items() 
            if column != "Outcome"
        ]
        condition = " & ".join(condition_parts)

        satisfying_idxs = df_origin_2.loc[test_idx].query(condition).index

        rule_violation_check[h] = (satisfying_idxs, outcome)

    input_for_test = {
        'data': df_tmp,
        'rule': rule_violation_check
    }

    # モデルの学習（提案モデル）----------------------------------------
    input_luka = {
        'L': L,
        'U': U,
        'S': S,
        'len_j': len_j,
        'len_l': len_l,
        'len_u': len_u,
        'len_s': len_s,
        'len_h': len_h,
        'len_i': len_i,
        'c1': settings['c1'],
        'c2': settings['c2'],
        'KB_origin': KB_origin,
        'target_predicate': 'Outcome',
        # 'kernel_function': linear_kernel,
        'kernel_function': partial(rbf_kernel, gamma=0.1),
    }

    problem_instance = Setup(input_luka, ObjectiveFunction)
    objective_function, constraints = problem_instance.main()
    problem = cp.Problem(objective_function, constraints)
    result = problem.solve(verbose=True)

    # テスト --------------------------------------------------------
    X_test = input_for_test['data'].drop(['target'], axis=1)
    y_test = input_for_test['data']['target']

    problem_info = problem_instance.problem_info # input_luka
    p_trained = Predicate_dual(problem_info, metrics="f1")
    # p_trained = Predicate_dual(problem_info, metrics="accuracy")
    y_pred = p_trained(X_test)
    y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

    # # 予測精度
    # result = {}
    # accuracy  = accuracy_score(y_test, y_pred_interpreted)
    # precision = precision_score(y_test, y_pred_interpreted)
    # recall    = recall_score(y_test, y_pred_interpreted)
    # f1        = f1_score(y_test, y_pred_interpreted)
    # auc       = roc_auc_score(y_test.replace(-1, 0), y_pred)

    # result['accuracy']  = float(accuracy)
    # result['precision'] = float(precision)
    # result['recall']    = float(recall)
    # result['f1']        = float(f1)
    # result['auc']       = float(auc)

    # print()
    # print()
    # print(f'accuracy: {accuracy}')
    # print(f'precision: {precision}')
    # print(f'recall: {recall}')
    # print(f'f1: {f1}')
    # print(f'auc: {auc}')
    # print()
    # print(confusion_matrix(y_test, y_pred_interpreted))

    # # ルール違反
    # result_rule_violation = {}
    # y_pred_interpreted = pd.DataFrame(y_pred_interpreted, index=test_idx)

    # for h, (idxs, ans) in input_for_test['rule'].items():

    #     violation_num  = int((y_pred_interpreted.loc[idxs] != ans).sum().iloc[0])
    #     violation_bool = 1 if violation_num >= 1 else 0
    #     result_rule_violation[h] = violation_bool

    # result['n_violation'] = sum(list(result_rule_violation.values()))
    # result['n_rule'] = len(result_rule_violation)
    # result['violation_rate'] = result['n_violation'] / result['n_rule']

    # settings['result'][f'fold_{i}'] = result

    result = evaluate_model(
        y_test,
        y_pred,
        y_pred_interpreted,
        input_for_test,
        test_idx
    )

    settings['result'][f'fold_{i}'] = result


# 実験結果の保存 ------------------------------------------------
with open(f'result.json', 'w') as f:
    json.dump(settings, f, indent=4)
    