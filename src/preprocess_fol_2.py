from typing import List, Dict, Tuple, Union
import sympy as sp
import pandas as pd
import numpy as np
from .operators import negation
from .misc import process_neg, is_symbol



# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    # def __init__(self):
    #     pass
    
    # FOLConverter の簡易動作確認用
    def __init__(self):
        self.len_j = 3
        self.len_u = 6
        


