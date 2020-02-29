import numpy as np
import skimage.measure

def Pooling(array):
    if len(array.shape) > 2:
        return np.array([Pooling(x)  for x in array])
    return MAXPooling(array)


def MAXPooling(Array,activation=1, ksize=2):
    assert len(Array) % ksize == 0

    return skimage.measure.block_reduce(Array, 
            (ksize,ksize), np.max)

