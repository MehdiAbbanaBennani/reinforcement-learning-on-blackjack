import numpy as np


def rmse(array1, array2):
    return np.sum(np.square(array1 - array2))