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

    for h, (idxs, ans) in input_for_test['rule'].items():

        violation_num  = int((y_pred_interpreted.loc[idxs] != ans).sum().iloc[0])
        violation_bool = 1 if violation_num >= 1 else 0
        result_rule_violation[h] = violation_bool

    result['n_violation'] = sum(list(result_rule_violation.values()))
    result['n_rule'] = len(result_rule_violation)
    result['violation_rate'] = result['n_violation'] / result['n_rule']

    return result