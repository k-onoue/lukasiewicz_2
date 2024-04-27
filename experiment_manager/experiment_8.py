import sys
project_dir_path = "/Users/keisukeonoue/ws/lukasiewicz_2"
sys.path.append(project_dir_path)

import json
import os
from functools import partial

import cvxpy as cp
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel

from src.misc import is_symbol
from src.setup_problem_dual_single_task import Setup
from src.setup_problem_primal import SetupPrimal
from src.objective_function_single_task import ObjectiveFunction
from src.predicate_single_task import Predicate_dual
from src.evaluation import evaluate_model

from sklearn.ensemble import RandomForestClassifier
from src.rulefit import RuleFitClassifier
from src.rulefit import ArrangeRules


project_dir_path = "/Users/keisukeonoue/ws/lukasiewicz_2"
data_dir_path = os.path.join(project_dir_path, "data/pima_indian_diabetes")

# 入力ファイル
file_path_1 = os.path.join(data_dir_path, "diabetes_cleaned_normalized.csv")
file_path_2 = os.path.join(data_dir_path, "diabetes_discretized.csv")

# # 入力ファイル
# file_path_1 = "data/pima_indian_diabetes/diabetes_cleaned_normalized.csv"
# file_path_2 = "data/pima_indian_diabetes/diabetes_discretized.csv"
# # file_path_3 = "data/pima_indian_diabetes/rules_3.txt"


df_origin_1 = pd.read_csv(file_path_1, index_col=0).reset_index(drop=True)
X_origin_1  = df_origin_1.drop(["Outcome"], axis=1)
y_origin_1  = df_origin_1["Outcome"]

df_origin_2 = pd.read_csv(file_path_2, index_col=0).reset_index(drop=True)
X_origin_2  = df_origin_2.drop(["Outcome"], axis=1)
y_origin_2  = df_origin_2["Outcome"]
print(df_origin_1.head())
print(df_origin_2.head())



# 実験設定
settings_list = [
    {
        'path': './experiments/version_81',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 0.1,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_82',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 1,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_83',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 5,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_84',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 15,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_85',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 50,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_86',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 100,
        'rule_thr': 0,
        'result': {}
    },
    {
        'path': './experiments/version_87',
        # 'source_paths': [file_path_1, file_path_2, file_path_3],
        'source_paths': [file_path_1, file_path_2],
        'experiment_name': 'pima_indian_diabetes_cv_8',
        'seed': 42,
        'n_splits': 5,
        'n_unsupervised': 15,
        'c1': 10,
        'c2': 1000,
        'rule_thr': 0,
        'result': {}
    },
]

for settings in settings_list:

    if not os.path.exists(settings['path']):
        os.makedirs(settings['path'])
        os.makedirs(os.path.join(settings['path'], "rules"))
        os.makedirs(os.path.join(settings['path'], "predictions"))


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


        # ルールの獲得 (RuleFit Classifier (discrete)）----------------------------------------
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
        pd.DataFrame(y_pred_interpreted, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/RuleFit Classifier (disc)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/RuleFit Classifier (disc)_{i}_proba.csv'))


        # ルールの整形 -------------------------------------------
        # rules_df = model.get_rules(exclude_zero_coef=True)
        rules_df = model.get_rules()
        rules_df.to_csv(os.path.join(settings['path'], f'rules/rules_{i}_original.csv'))
        rules_df = rules_df[rules_df['coef'].abs() > settings['rule_thr']]
        rules_df.to_csv(os.path.join(settings['path'], f'rules/rules_{i}.csv'))

        if rules_df.shape[0] == 0:
            print("There is no rule!")
            continue

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

        
        # モデルのテスト 4 (RuleFit Classifier (discrete)）
        result = evaluate_model(
            pd.DataFrame(y_test, index=test_idx),
            pd.DataFrame(y_pred, index=test_idx),
            pd.DataFrame(y_pred_interpreted, index=test_idx),
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['RuleFit Classifier (disc)'] = result

        # tree generator
        y_pred_interpreted = model.tree_generator.predict(X_test)
        y_pred = model.tree_generator.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/tree generator (disc)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/tree generator (disc)_{i}_proba.csv'))

        result = evaluate_model(
            pd.DataFrame(y_test, index=test_idx),
            pd.DataFrame(y_pred, index=test_idx),
            pd.DataFrame(y_pred_interpreted, index=test_idx),
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['tree generator (disc)'] = result

        # モデルの学習とテスト 9, 10 (RuleFit Classifier (continuous)）----------------------------------------
        from sklearn.ensemble import RandomForestClassifier
        from src.rulefit import RuleFitClassifier
        X_train = X_origin_1.copy().iloc[train_idx].values
        y_train = y_origin_1.copy().iloc[train_idx].values
        X_test  = X_origin_1.copy().iloc[test_idx].values
        y_test  = y_origin_1.copy().iloc[test_idx].values

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
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/RuleFit Classifier (conti)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/RuleFit Classifier (conti)_{i}_proba.csv'))

        result = evaluate_model(
            pd.DataFrame(y_test, index=test_idx),
            pd.DataFrame(y_pred, index=test_idx),
            pd.DataFrame(y_pred_interpreted, index=test_idx),
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['RuleFit Classifier (conti)'] = result

        # tree generator
        y_pred_interpreted = model.tree_generator.predict(X_test)
        y_pred = model.tree_generator.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/tree generator (conti)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/tree generator (conti)_{i}_proba.csv'))

        result = evaluate_model(
            pd.DataFrame(y_test, index=test_idx),
            pd.DataFrame(y_pred, index=test_idx),
            pd.DataFrame(y_pred_interpreted, index=test_idx),
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['tree generator (conti)'] = result

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

        # # ルール
        # KB_origin =  []

        # with open(file_path_3, 'r') as file:
        #     for line in file:
        #         formula = line.split()
        #         KB_origin.append(formula)
        # ルール
        KB_origin = KB_origin

        # パラメータ
        len_j = len(L)
        len_l = len(train_idx)
        len_u = settings['n_unsupervised']
        len_s = len_l + len_u

        len_h = len(KB_origin)
        len_i = len_u * 2


        # モデルの学習 4（提案モデル）----------------------------------------
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

        # テスト --------------------------------------------------------
        X_test = input_for_test['data'].drop(['target'], axis=1)
        y_test = input_for_test['data']['target']

        problem_info = problem_instance.problem_info # input_luka
        p_trained = Predicate_dual(problem_info, metrics="f1")
        # p_trained = Predicate_dual(problem_info, metrics="accuracy")
        y_pred = p_trained(X_test)
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/linear svm (L)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/linear svm (L)_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['linear svm (L)'] = result

        # モデルの学習 5（提案モデル）----------------------------------------
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

        # テスト --------------------------------------------------------
        X_test = input_for_test['data'].drop(['target'], axis=1)
        y_test = input_for_test['data']['target']

        problem_info = problem_instance.problem_info # input_luka
        p_trained = Predicate_dual(problem_info, metrics="f1")
        # p_trained = Predicate_dual(problem_info, metrics="accuracy")
        y_pred = p_trained(X_test)
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/non-linear svm (L)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/non-linear svm (L)_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['non-linear svm (L)'] = result

        # モデルの学習 6（提案モデル）----------------------------------------
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
            'kernel_function': "logistic regression",
        }

        problem_instance = SetupPrimal(input_luka_1)
        objective_function, constraints = problem_instance.main()
        problem = cp.Problem(objective_function, constraints)
        result = problem.solve(verbose=True)

        # テスト --------------------------------------------------------
        X_test = input_for_test['data'].drop(['target'], axis=1)
        y_test = input_for_test['data']['target']

        problem_info = problem_instance.problem_info # input_luka
        p_name = problem_instance.problem_info['target_predicate']
        p_trained = problem_instance.problem_info['predicates_dict'][p_name]

        y_pred = p_trained(X_test).value
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/logistic regression (L)_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/logistic regression (L)_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['logistic regression (L)'] = result


        # モデルの学習とテスト 1（linear svm）----------------------------------------
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV
        X_train = X_origin_1.copy().iloc[train_idx]
        y_train = y_origin_1.copy().iloc[train_idx]
        X_test  = X_origin_1.copy().iloc[test_idx]
        y_test  = y_origin_1.copy().iloc[test_idx]

        linear_svm = SVC(kernel='linear')
        model = CalibratedClassifierCV(linear_svm)
        model.fit(X_train, y_train)

        y_pred_interpreted = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/linear svm_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/linear svm_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['linear svm'] = result


        # モデルの学習とテスト 2 (non-linear svm) --------------------------------------------------------
        from sklearn.svm import SVC
        X_train = X_origin_1.copy().iloc[train_idx]
        y_train = y_origin_1.copy().iloc[train_idx]
        X_test  = X_origin_1.copy().iloc[test_idx]
        y_test  = y_origin_1.copy().iloc[test_idx]

        model = SVC(kernel='rbf', gamma=0.1, probability=True)
        model.fit(X_train, y_train)

        y_pred_interpreted = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/non-linear svm_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/non-linear svm_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['non-linear svm'] = result

    
        # モデルの学習とテスト 3 (logistic regression) --------------------------------------------------------
        from sklearn.linear_model import LogisticRegression
        X_train = X_origin_1.copy().iloc[train_idx]
        y_train = y_origin_1.copy().iloc[train_idx]
        X_test  = X_origin_1.copy().iloc[test_idx]
        y_test  = y_origin_1.copy().iloc[test_idx]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred_interpreted = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/logistic regression_{i}.csv'))
        pd.DataFrame(y_pred, index=test_idx).to_csv(os.path.join(settings['path'], f'predictions/logistic regression_{i}_proba.csv'))

        result = evaluate_model(
            y_test,
            y_pred,
            y_pred_interpreted,
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['logistic regression'] = result


        # ルールの多数決による予測

        df_rule_based_pred = pd.DataFrame(columns=['rule_ids', 'preds', 'pred_tmp', 'pred'])

        for l in test_idx:
            rule_ids = []
            preds = []
            pred_tmp = 0

            for h, idx_outcome in rule_violation_check.items():
                if l in idx_outcome[0]:
                    rule_ids.append(h)
                    preds.append(idx_outcome[1])

                    pred_tmp += idx_outcome[1]
                
            df_rule_based_pred.loc[l, 'rule_ids'] = rule_ids
            df_rule_based_pred.loc[l, 'preds'] = preds
            df_rule_based_pred.loc[l, 'pred_tmp'] = pred_tmp

            # 多数決で引き分けの場合はどうすべきか
            # 適用可能ルールが無いときは陰性（-1）
            # 多数決で引き分けの際は陰性（-1）
            if pred_tmp > 0:
                df_rule_based_pred.loc[l, 'pred'] = 1
            else:
                df_rule_based_pred.loc[l, 'pred'] = -1

        df_rule_based_pred.to_csv(os.path.join(settings['path'], f'rules/rule_based_pred_{i}.csv'))
            
        y_pred = df_rule_based_pred['pred'].replace(-1, 1).values
        y_pred_interpreted = df_rule_based_pred['pred'].replace(-1, 1).values

        result = evaluate_model(
            pd.DataFrame(y_test, index=test_idx),
            pd.DataFrame(y_pred, index=test_idx),
            pd.DataFrame(y_pred_interpreted, index=test_idx),
            input_for_test,
            test_idx
        )

        settings['result'][f'fold_{i}']['rule_based_prediction'] = result


    # 実験結果の保存 -----------------------------------------------
    with open(os.path.join(settings['path'], 'result.json'), 'w') as f:
        json.dump(settings, f, indent=4)
        


