# Eyeriss Chip simulator
This is a Eyeriss chip simulator that does the same thing as Eyeriss. It was inspired by [EyerissF](https://github.com/jneless/Eyerissf). 

However, EyerissF is not a full simulator as it does integrate full Eyeriss mapping strategy into consideration. Also, it does not test a full images but one picture. If you try with the complete minist dataset, EyerissF is not working.

This repo reimplemented and reorganized all functions of EyerissF.

## How does Eyeriss Work

Eyeriss is a row stationary DNN accelerator. 

If this is the first time you heard about Eyeriss, or you are not very familiar with Eyeriss, the [readme](https://github.com/jneless/EyerissF/blob/master/README.md) of [EyerissF](https://github.com/jneless/Eyerissf) gives a good explanation. Please refer to that.

## File Structure

### [Source](src/)

* Configuration
    * [conf.py](src/conf.py) ( default Eyeriss config features)
* Eyeriss Chip
    * [PE.py](PE.py) ( row stationary processing element)
    * [EyerissF.py](EyerissF.py) (containing 168 PEs; mapping a *Pass* of [*Weight, IfMap*] to each PE)
* Hive
    * [Conv2d](src/Hive.py)
    * [FullyConnect](src/Hive.py)
    * [Activiation](src/Activiation.py)
    * [Pooling](src/Pooling.py)
    * [Pre/PostProcess](src/IO2.py) ( Compress and Decompress)

### [Test](test/)
* [LeNet5_Hive.py](test/Lenet5_Hive.py) (test LeNet5 on Mnist, output inference result)
* [test_IO2.py](test/test_IO2.py) (test compression and decompression)
