import sys
sys.path.append('../')
from src import IO2
import numpy as np
r = IO2.RLE(1)
a=np.arange(16).reshape(2,2,-1)
b=r.Compress(a)
c=r.Decompress(b)
assert np.all(a == c)
print('before compression:', a)
print('after compression:', b)
print('after depression:', c)

