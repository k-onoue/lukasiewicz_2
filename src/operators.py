# from typing import List, Union
import sympy as sp

# def negation(x: Union[List[Union[str, cp.Expression]], Union[str, cp.Expression]]) -> Union[]:
def negation(x, is_1_symbol=False):
    """
    ¬ Negation:
    This function is used in the following two cases:
    1. Take the negation of predicate(x)
    2. Take the negation of a formula (list)
    
    Parameters
    ----------
    x : Union[List[Union[str, sp.Expr]], Union[str, sp.Expr]]
        The input to be negated, either a single predicate or a list representing a formula.
    is_1_symbol : bool, optional
        A flag indicating whether `x` is a single symbol. Default is False.
    
    Returns
    -------
    formula : Union[int, List[Union[str, sp.Expr]]]
        The negated value or formula.
    
    Raises
    ------
    ValueError
        If the input formula contains '→'.
    
    Examples
    --------
    >>> negation('x')
    '¬x'
    
    >>> negation(['x', '∧', 'y'])
    ['¬x', '∨', '¬y']
    
    Notes
    -----
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
