import json
import os
from functools import partial

import cvxpy as cp
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel

from src.misc import is_symbol
from src.setup_problem_dual_single_task import Setup
from src.objective_function_single_task import ObjectiveFunction
from src.predicate_single_task import Predicate_dual
from src.evaluation import evaluate_model

from sklearn.ensemble import RandomForestClassifier
from src.rulefit import RuleFitClassifier
from src.rulefit import ArrangeRules



# 入力ファイル
file_path_1 = "data/pima_indian_diabetes/diabetes_cleaned_normalized.csv"
file_path_2 = "data/pima_indian_diabetes/diabetes_discretized.csv"
# file_path_3 = "data/pima_indian_diabetes/rules_3.txt"


df_origin_1 = pd.read_csv(file_path_1, index_col=0).reset_index(drop=True)
X_origin_1  = df_origin_1.drop(["Outcome"], axis=1)
y_origin_1  = df_origin_1["Outcome"]

df_origin_2 = pd.read_csv(file_path_2, index_col=0).reset_index(drop=True)
X_origin_2  = df_origin_2.drop(["Outcome"], axis=1)
y_origin_2  = df_origin_2["Outcome"]
print(df_origin_1.head())
print(df_origin_2.head())



# # 実験設定
# settings = {
#     'path': './experiments/version_3',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 10,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_4',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 20,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_5',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 50,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_6',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 100,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_7',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 1000,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_8',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 5,
#     'result': {}
# }

# # 実験設定
# settings = {
#     'path': './experiments/version_9',
#     # 'source_paths': [file_path_1, file_path_2, file_path_3],
#     'source_paths': [file_path_1, file_path_2],
#     'experiment_name': 'pima_indian_diabetes_cv_3',
#     'seed': 42,
#     'n_splits': 5,
#     'n_unsupervised': 1,
#     'c1': 10,
#     'c2': 1,
#     'result': {}
# }

# 実験設定
settings = {
    'path': './experiments/version_10',
    # 'source_paths': [file_path_1, file_path_2, file_path_3],
    'source_paths': [file_path_1, file_path_2],
    'experiment_name': 'pima_indian_diabetes_cv_3',
    'seed': 42,
    'n_splits': 5,
    'n_unsupervised': 1,
    'c1': 10,
    'c2': 0.1,
    'result': {}
}


if not os.path.exists(settings['path']):
    os.makedirs(settings['path'])
    os.makedirs(os.path.join(settings['path'], "rules"))


kf = KFold(n_splits=settings['n_splits'])

idx_split = {}

for i, (train_idx, test_idx) in enumerate(kf.split(df_origin_1)):

    print()
    print()
    print()
    print()
    print()
    print(f"fold: {i+1} of {settings['n_splits']}")

    settings['result'][f'fold_{i}'] = {}


    idx_split[i] = train_idx.tolist(), test_idx.tolist()


    # ルールの獲得 (RuleFit Classifier (continuous)）----------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    from src.rulefit import RuleFitClassifier
    from src.rulefit import ArrangeRules
    X_train = X_origin_2.copy().iloc[train_idx].values
    y_train = y_origin_2.copy().iloc[train_idx].values
    X_test  = X_origin_2.copy().iloc[test_idx].values
    y_test  = y_origin_2.copy().iloc[test_idx].values

    feature_names = list(X_origin_2.columns)

    model = RuleFitClassifier(
        rfmode='classify',
        tree_generator=RandomForestClassifier(random_state=42),
        random_state=42,
        exp_rand_tree_size=False
    )

    model.fit(X_train, y_train, feature_names=feature_names)

    y_pred_interpreted = model.predict(X_test)
    y_pred = model.predict_proba(X_test)[:, 1]


    # ルールの整形 -------------------------------------------
    rules_df = model.get_rules(exclude_zero_coef=True)
    rule_processor = ArrangeRules(
        rules_df,
        feature_names=feature_names,
        conclusion_name="Outcome"
    )
    KB_origin = rule_processor.construct_KB()
    rule_processor.save_KB_as_txt(os.path.join(settings['path'], f'rules/rules_{i}.txt'))


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

    # テストデータ -------------------------------------------------------
    df_tmp = df_origin_1.copy().iloc[test_idx, :]
    df_tmp= df_tmp.rename(columns={'Outcome': 'target'})
    df_tmp['target'] = df_tmp['target'].replace(0, -1)

    input_for_test = {
        'data': df_tmp,
        'rule': rule_violation_check
    }


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

    # パラメータ
    len_j = len(L)
    len_l = len(train_idx)
    len_u = settings['n_unsupervised']
    len_s = len_l + len_u

    len_h = len(KB_origin)
    len_i = len_u * 2


    # モデルの学習 1（提案モデル）----------------------------------------
    input_luka_1 = {
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
        'kernel_function': linear_kernel,
    }

    problem_instance = Setup(input_luka_1, ObjectiveFunction)
    objective_function, constraints = problem_instance.main()
    problem = cp.Problem(objective_function, constraints)
    result = problem.solve(verbose=True)

    # テスト 1 --------------------------------------------------------
    X_test = input_for_test['data'].drop(['target'], axis=1)
    y_test = input_for_test['data']['target']

    problem_info = problem_instance.problem_info # input_luka
    p_trained = Predicate_dual(problem_info, metrics="f1")
    # p_trained = Predicate_dual(problem_info, metrics="accuracy")
    y_pred = p_trained(X_test)
    y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

    result = evaluate_model(
        y_test,
        y_pred,
        y_pred_interpreted,
        input_for_test,
        test_idx
    )

    settings['result'][f'fold_{i}']['linear svm (L)'] = result

    # モデルの学習 2（提案モデル）----------------------------------------
    input_luka_1 = {
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
        'kernel_function': partial(rbf_kernel, gamma=0.1),
    }

    problem_instance = Setup(input_luka_1, ObjectiveFunction)
    objective_function, constraints = problem_instance.main()
    problem = cp.Problem(objective_function, constraints)
    result = problem.solve(verbose=True)

    # テスト 2 --------------------------------------------------------
    X_test = input_for_test['data'].drop(['target'], axis=1)
    y_test = input_for_test['data']['target']

    problem_info = problem_instance.problem_info # input_luka
    p_trained = Predicate_dual(problem_info, metrics="f1")
    # p_trained = Predicate_dual(problem_info, metrics="accuracy")
    y_pred = p_trained(X_test)
    y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

    result = evaluate_model(
        y_test,
        y_pred,
        y_pred_interpreted,
        input_for_test,
        test_idx
    )

    settings['result'][f'fold_{i}']['non-linear svm (L)'] = result


# 実験結果の保存 -----------------------------------------------
with open(os.path.join(settings['path'], 'result.json'), 'w') as f:
    json.dump(settings, f, indent=4)
    