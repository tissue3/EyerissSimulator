from Hive import Hive
from EyerissF import EyerissF as EF
import numpy as np
import Extension
import skimage.io as io
from skimage.util import pad
import os


dir_name = "mnist_png/mnist_png/training/2"
#dir_name = "mnist_png/mnist_png/one_pic"
files = os.listdir(dir_name)

ef = EF()
hive = Hive(ef)
for f in files:
    load_from = os.path.join(dir_name,f)
    image = io.imread(load_from, as_gray=True)
    image = pad(image,((2,2),(2,2)), 'median')
    pics = np.array(image.astype(int)).reshape(1,1,image.shape[0],-1)
    
    flts = np.float16(np.load("filter/convnet.c1.weight.npy"))
    pics= hive.Conv2d(pics,flts)
    pics = hive.ReLU(pics)
    pics=hive.Pooling(pics,255)
    
    flts = np.float16(np.load("filter/convnet.c3.weight.npy"))
    print(pics.shape, filts.shape)
    r = hive.Conv2d(pics,flts)
    #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.ReLU(pics)
    pics=hive.Pooling(pics,255)
    
    flts = np.float16(np.float16(np.load("filter/convnet.c3.weight.npy")))
    r = hive.Conv2d(pics, flts) 
    #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.ReLU(r)


    vector = hive.FullConnect(vector, np.load('filter/fc.f6.weight.npy'))

    vector = hive.FullConnect(vector, np.load('filter/fc.f7.weight.npy'))

    print("this number is : ",vector.argmax())
