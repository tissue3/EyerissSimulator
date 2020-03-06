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

### Hive platform

Hive is a new CNN platform based on Eyeriss chip or EyerissF simulator, which contains basic funxs to establish CNN.
**(NOT "APACHE HIVE")**
Eyeriss or EyerissF just only an ASIC chip and python-made simulator and can not achieve any tasks. In order to do pattern regonization tasks, it must have a mature platform to support standard input data.
Hive is aiming to tranfor 3-channel jpg pics to input Eyeriss supported stream and decompress results.


If u wanna use hive to create CNN, u should do following steps:

1. init EyerissF simulator 
```python
ef = EyerissF()
```
or

```python
ef = EyerissF("manual")
```

1. init Hive
```python
hive = Hive(ef)
```

Other funxs from Hive
1. convolution ( Eyeriss Supported)
```python
hive.conv2d(pics,filters,number of pics, number of filters)
```
1. Pooling
```python
hive.Pooling(pics)
```
1. FullConnect
```python
hive.FullConnect(vector,vector2)
```


## contact me
please Email lee@frony.net or k1924116@kcl.ac.uk

## REALLY IMPORTANT
**Emergy model IS NOT finished YET**

## last updated
2019 Sep 23th

## __future__
1. changing name in case of any misunderstanding with 'Apache Hive'
1. overriding file constructions to make basic code in src folder
1. add new functions in energy stream calculation between different storage layers 
