def ReLU1D(array):
    array[array < 0] = 0
    return array

def ReLU(array):
    if len(array.shape) > 1
        return np.array([ReLU(x)  for x in array])
    else:
        return ReLU1D(array)
