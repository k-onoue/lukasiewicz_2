# from typing import List, Union
import sympy as sp


# def negation(x: Union[List[Union[str, cp.Expression]], Union[str, cp.Expression]]) -> Union[]:
def negation(x, is_1_symbol=False):
    """
    ¬ 否定:
    この関数は以下の 2 つの場合に使用する．
    1. predicate(x) の否定を取る
    2. formula (list) の否定を取る 
    """
    if type(x) != list:
        if not is_1_symbol:
            return 1 - x
        else:
            return sp.Symbol('1') - x
    else:
        formula = []
        flag_neg = False
        for i, item in enumerate(x):
            if item == '∧':
                formula.append('∨')
            elif item == '∨':
                formula.append('∧')
            elif item == '⊕':
                formula.append('⊗')
            elif item == '⊗':
                formula.append('⊕')
            elif item == '¬':
                flag_neg = True
                pass
            elif item == '→':
                print("This may cause an error, please eliminate '→' first.")
            # else:
            #     # if x[i-1] == '¬':
            #     #     formula.append(item)
            #     # else:
            #     #     formula.append('¬')
            #     #     formula.append(item)
            #     formula.append('¬')
            #     formula.append(item)

            else:
                if flag_neg == False:
                    formula.append('¬')
                    formula.append(item)
                else:
                    formula.append(item)
                    flag_neg = False

        return formula
