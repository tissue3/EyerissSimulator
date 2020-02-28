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
    pics = np.swapaxes(pics,1,3)
    pics= pics+np.float16(np.load("filter/convnet.c1.bias.npy"))
    pics = np.swapaxes(pics,1,3)
    pics = hive.ReLU(pics)
    pics=hive.Pooling(pics,255)
    
    flts = np.float16(np.load("filter/convnet.c3.weight.npy"))
    print('pic', pics.shape,'flt', flts.shape)
    pics = hive.Conv2d(pics,flts)
    pics = np.swapaxes(pics,1,3)
    pics= pics+np.float16(np.load("filter/convnet.c3.bias.npy"))
    pics = np.swapaxes(pics,1,3)
    #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.ReLU(pics)
    pics=hive.Pooling(pics,255)
    
    print('after pooling pic', pics.shape)
    
    flts = np.float16(np.float16(np.load("filter/convnet.c5.weight.npy")))
    print('pic', pics.shape,'flt', flts.shape)
    pics = hive.Conv2d(pics, flts) 
    pics = np.swapaxes(pics,1,3)
    pics= pics+np.float16(np.load("filter/convnet.c5.bias.npy"))
    pics = np.swapaxes(pics,1,3)
    #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
    pics = hive.ReLU(pics)

    vector = pics.flatten()
    vector = hive.FullConnect(vector, np.load('filter/fc.f6.weight.npy'))
    vector = vector+np.float16(np.load("filter/fc.f6.bias.npy"))
    vector = hive.FullConnect(vector, np.load('filter/fc.f7.weight.npy'))
    vector = vector+np.float16(np.load("filter/fc.f7.bias.npy"))

    print("this number is : ",vector.argmax())
