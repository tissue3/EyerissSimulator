import numpy as np

def Pooling(array,activation):
    if len(array.shape) > 2:
        return np.array([Pooling(x, activation)  for x in array])
    return MAXPooling(array,activation)


def MAXPooling(Array,activation=1, ksize=2):
    print(Array.shape)
    assert len(Array) % ksize == 0

    V2list = np.vsplit(Array, len(Array) / ksize)

    VerticalElements = list()
    HorizontalElements = list()

    for x in V2list:
        H2list = np.hsplit(x, len(x[0]) / ksize)
        HorizontalElements.clear()
        for y in H2list:
            # y should be a two-two square
            HorizontalElements.append(y.max())
        VerticalElements.append(np.array(HorizontalElements))

    return np.array(np.array(VerticalElements)/activation,dtype=int)
