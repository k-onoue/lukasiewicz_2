import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def evaluate_model(
    y_test,
    y_pred,
    y_pred_interpreted,
    input_for_test: dict,
    test_idx
) -> dict:
    # 予測精度
    result = {}
    accuracy  = accuracy_score(y_test, y_pred_interpreted)
    precision = precision_score(y_test, y_pred_interpreted)
    recall    = recall_score(y_test, y_pred_interpreted)
    f1        = f1_score(y_test, y_pred_interpreted)
    auc       = roc_auc_score(y_test.replace(-1, 0), y_pred)

    result['accuracy']  = float(accuracy)
    result['precision'] = float(precision)
    result['recall']    = float(recall)
    result['f1']        = float(f1)
    result['auc']       = float(auc)

    print()
    print()
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    print(f'auc: {auc}')
    print()
    print(confusion_matrix(y_test, y_pred_interpreted))

    # ルール違反
    result_rule_violation = {}
    y_pred_interpreted = pd.DataFrame(y_pred_interpreted, index=test_idx)


    result['violation_detail'] = {}

    evaluation_num_instance = 0
    violation_num_instance = 0

    for h, (idxs, ans) in input_for_test['rule'].items():

        evaluation_num = len(idxs)
        df_evaluated = y_pred_interpreted.loc[idxs] != ans
        violating_indexes = df_evaluated[df_evaluated[0] == True].index.tolist()
        violation_num = len(violating_indexes)

        evaluation_num_instance += evaluation_num
        violation_num_instance += violation_num

        result['violation_detail'][h] = (evaluation_num, violation_num , violating_indexes)


        # violation_num  = int((y_pred_interpreted.loc[idxs] != ans).sum().iloc[0])
        violation_bool = 1 if violation_num >= 1 else 0
        result_rule_violation[h] = violation_bool

    result['n_violation'] = sum(list(result_rule_violation.values()))
    result['n_rule'] = len(result_rule_violation)
    result['violation_rate'] = result['n_violation'] / result['n_rule']

    result['n_violation (instance)'] = violation_num_instance
    result['n_evaluation (instance)'] = evaluation_num_instance
    result['violation_rate (instance)'] = violation_num_instance / evaluation_num_instance

    return result