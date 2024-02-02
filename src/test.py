import numpy as np 


def mk_arr():
    arr = np.random.rand(5, 5)
    arr = arr @ arr
    return arr

