import numpy as np
import conf
import IO2
import Pooling
import Activiation
import utils

class Hive():

    def __init__(self, EyerissF, mode="auto"):
        self.mode = mode
        self.EyerissF = EyerissF
        self.RLE = IO2.RLE(RateNeed = 0)
        
        #maybe wrap it to a separate class
        self.m = 1
        self.n = 1
        self.e = 1
        self.p = 1
        self.q = 1
        self.r = 1
        self.t = 1
        
    def Conv2d(self, Pictures=0, FilterWeights=0, PictureNum=0, FilterWeightNum=0, Channels=0):
        if self.mode == "auto":

            # auto mode should compress data inside
            # TODO: DRAM read
            Pictures = self.RLE.Compress(Pictures)
            FilterWeights = self.RLE.Compress(FilterWeights)
            self.input(Pictures, FilterWeights, PictureNum, FilterWeightNum, Channels)
            self.Conv2LogicalMapping()
            self.Conv2PhysicalMapping()
            self.mode = "manual"
            self.Conv2d(0, 0, 0, 0)
            self.mode = "auto"
            self.Reverse()
            return self.Output()
        else:
            self.t = [self.EyerissF.Conv2d(x, self.FilterWeight, self.PictureNum, self.FilterWeightNum) for x in self.mapping]
            self.TempPsum = np.vstack(self.t)
            
    def Relu(self, array):
        if type(array) == type(list()):
            return Activiation.ReluArray(array)
        else:
            return Activiation.Relu(array)

    def Pooling(self, array, activation=1):
        if type(array) == type(list()):
            return Pooling.Pooling(array, activation)
        else:
            return Pooling.MAXPooling(array, activation)
            
    def input(self, Pictures, FilterWeights, PictureNum, FilterWeightNum, Channels):
        # input fiture map, filter/weights, batch size, filter number, channel number

        Pictures = self.RLE.Decompress(Pictures)
        FilterWeights = self.RLE.Decompress(FilterWeights)

        self.Pictures = Pictures
        self.FilterWeights = FilterWeights
        self.PictureNum = PictureNum
        self.FilterWeightNum = FilterWeightNum
        self.Channels = Channels

    def Conv2LogicalMapping(self):
        self.__PEArrayMapping__()
        self.__PESetMapping__()
        if self.n != 1: self.__FmapReuse__()
        if self.p != 1: self.__FilterReuse__()
        if self.q != 1: self.__ChannelAccumulation__()

    def Conv2PhysicalMapping(self):

        FilterWeight = self.FilterWeight
        Picture = self.Picture
        FilterWeight = self.FilterWeight
        x = 0
        t = list()
        while conf.EyerissWidth * x + conf.EyerissWidth + len(FilterWeight) - 1 < len(FilterWeight) + len(Picture) - 1:
            P = Picture[conf.EyerissWidth * x: conf.EyerissWidth * x + conf.EyerissWidth + len(FilterWeight) - 1]
            x = x + 1
            t.append(P)

        P = Picture[conf.EyerissWidth * x:]

        # 判断逻辑矩阵的尾巴，并删除多余的图
        if len(Picture[conf.EyerissWidth * x:]) < len(FilterWeight):
            pass
        else:
            t.append(P)

        self.__SetPhysicalMapping__(t)

    def __SetPhysicalMapping__(self, mapping):
        self.mapping = mapping
    
    def __SetMappingParameters__(self, m=0, n=0, e=0, p=0, q=0, r=0, t=0):
        self.m = m if m!=0 else self.m
        self.n = n if n!=0 else self.n
        self.e = e if e!=0 else self.e
        self.p = p if p!=0 else self.p
        self.q = q if q!=0 else self.q
        self.r = r if r!=0 else self.r
        self.t = t if t!=0 else self.t
        
    
    def __PEArrayMapping__(self):
        #TODO: also consider stride
        PESetHeight = self.FilterWeights.shape[3] #filter height
        PESetWidth = self.Pictures.shape[3]- self.FilterWeights.shape[3] + 1 #ofmap height
        #Eyeriss only support filter height smaller than PE array height
        assert PESetHeight <= conf.EyerissHeight
        t = int(conf.EyerissHeight/PESetHeight) # t filters
        
        if PESetWidth > conf.EyerissWidth:
            #strip-mining the 2-D convolution
            # filter height 
            fold = ( int((PESetWidth-1)/conf.EyerissWidth) + 1 )
            e = conf.EyerissWidth
            if t%fold == 0:
                t = int(t/fold)
                e = PESetWidth
        else:
            e = PESetWidth
        self.__SetPicAndFlt__(e=e,t=t)
        
    def __PESetMapping__(self):
        #TODO: add reusing filter, processing n ifmaps at a time
        slidingWindow = self.Pictures.shape[4]
        q = int(conf.IfmapSpad/(slidingWindow*self.n))
        p = min(int(conf.PsumSpad/self.n), int(conf.FilterSpad/(q*slidingWindow))
        #TODO:channel, filter number should be divisible by q, p
        self.__SetPicAndFlt__(q=e,p=t)
        
        
    def __SetPicAndFlt__(self, Pictures=None, FilterWeights=None):
        self.Pictures = Pictures if Pictures !=None else self.Pictures
        self.FilterWeights = FilterWeights 
                      if FilterWeigths != None else self.FilterWeights
        
    def __FilterReuse__(self):
        assert self.n != 1 and self.Pictures.shape[0]%self.n == 0
        
        Pictures = np.split(self.Pictures, self.n)
        Pictures = np.concatenate(Pictures, axis = 2)
            
        self.__SetPicAndFlt__(Pictures=Pictures)

    def __FmapReuse__(self):
        assert self.p != 1 and self.FilterWeights.shape[0]%self.p == 0
        
        s = np.array(FilterWeights.shape)
        s[2] *= self.p
        FilterWeights = np.empty(s,dtype=FilterWeights.dtype)
        for p in range(self.p):
            FilterWeights[:,:, p::self.p,:] = self.FilterWeights[p]
        
        self.__SetPicAndFlt__(FilterWeights = FilterWeights)

    def __ChannelAccumulation__(self):
        assert self.q != 1 and self.FilterWeights.shape[1]%self.q == 0
        
        
        s = np.array(Pictures.shape)
        s[2] *= self.q
        Pictures = np.empty(s,dtype=Pictures.dtype)
        for q in range(self.q):
            Pictures[:,:, q::self.q,:] = self.Pictures[:,q,:,:]
            
        s = np.array(FilterWeights.shape)
        s[2] *= self.q
        FilterWeights = np.empty(s,dtype=FilterWeights.dtype)
        for q in range(self.q):
            FilterWeights[:,:, q::self.q,:] = self.FilterWeights[:,q,:,:]
    
        self.__SetPicAndFlt__(Picture, FilterWeights)
        

    def Reverse(self):
        Psum = self.TempPsum
        if self.PictureNum == 1 and self.FilterWeightNum == 1:
            self.__SetReturnImgs__(Psum)
        if self.PictureNum == 1:
            self.__ReverseFmapReuse__(Psum, self.FilterWeightNum)
        elif self.FilterWeightNum == 1:
            self.__ReverseFilterReuse__(Psum, self.PictureNum)

    def __ReverseFmapReuse__(self, Psum, PsumNum):
        SubMap = np.hsplit(Psum, int(np.shape(Psum)[1] / PsumNum))
        l = []
        m = []
        for x in range(0, PsumNum):
            for y in range(len(SubMap)):
                # [np.newaxis]会使返回的向量为列向量
                l.append(np.transpose(np.array(SubMap[y][:, x])[np.newaxis]))
            m.append(np.hstack(l))
            l = []

        self.__SetReturnImgs__(m)

    def __ReverseFilterReuse__(self, Psum, PsumNum):
        self.__SetReturnImgs__(list(np.hsplit(Psum, PsumNum)))

    def __SetReturnImgs__(self, ReturnImgs):
        self.ReturnImgs = ReturnImgs

    def Output(self):
        #TODO: trace memory write
        return self.Compress(self.ReturnImgs)

    def FullConnect(self, v1, v2, activation=1):
        return np.array(np.dot(v1, v2) / activation, dtype=int)
