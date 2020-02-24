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
    pics = [image.astype(int)]
    
    flts = np.load("filter/convnet.c1.weight.npy")

    pics= hive.Conv2d(pics,flts,1,6)
    pics = hive.Relu(pics)
    pics=hive.Pooling(hive.Decompress(pics),255)

    r = hive.Conv2d(pics, np.load("filter/convnet.c3.weight.npy"),6, 16)
    pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.Relu(pics)
    pics=hive.Pooling(pics,255)
    
    
    r = hive.Conv2d(pics, np.load("filter/convnet.c3.weight" + str(x) + ".npy"),16, 120) 
    pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.Relu(pics)


    vector = hive.FullConnect(np.array(pics).flatten(),np.load('FullConnectLayer1/FullConnectLayer1.npy'),255)
    pics = hive.Relu(pics)

    vector = hive.FullConnect(vector, np.load('FullConnectLayer1/FullConnectLayer2.npy'))

    vector = hive.FullConnect(vector, np.load('FullConnectLayer1/FullConnectLayer3.npy'))

    print("this number is : ",vector.argmax())
