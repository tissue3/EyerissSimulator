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
        self.GLB = conf.GLB
        #TODO: read/write communications between GLB, PEArray, individual is missing
        
        #maybe wrap it to a separate class
        self.m = 1
        self.n = 1
        self.p = 1
        self.q = 1
        self.e = 1
        self.r = 1
        self.t = 1
        
    def Conv2d(self, Pictures=0, FilterWeights=0, PictureNum=0, FilterWeightNum=0, Channels=0):
        Pictures = self.RLE.Compress(Pictures)
        FilterWeights = self.RLE.Compress(FilterWeights)
        Passes = self.input(Pictures, FilterWeights, PictureNum, FilterWeightNum, Channels)
        Psum = [self.EyerissF.Conv2d(ps, self.n, self.p, self.q) for ps in Passes]
        self.Reverse(Psum)
        return self.Output()
            
    def ReLU(self, array):
        return Activiation.ReLU(array)

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

    def Conv2DMapping(self):
        self.__PESetMapping__()
        self.__PEArrayMapping__()
        self.__SetPasses__()

    def __SetPasses__(self):
        Passes = []
        for batch in range( self.Pictures.shape[0] ):
            for ofmap in range( int(self.FilterWeights.shape[0]/self.t) ):
                for channel in range( int(self.Pictures.shape[1]/self.r) ):
                    #TODO: let's assume there is no reuse of filter vertically
                    #self.e can be smaller than actual ifmapWidth
                    #e.g. self.e of 27 is sent out, so 14 for one and 13 for another pass
                    #for outmapwidth of 14, filterwidth of 2, we need 14+2-1 ifmaps
                    ifmapEachPass = self.e + self.FilterWeights.shape[2] - 1
                    ofmapWidth = self.Pictures.shape[2] - self.FilterWeigths.shape[2] + 1
                    head = 0
                    for e in range( int(ofmapWidth/self.e) ):
                        tail = min(conf.EyerissWidth*(e+1), head+ifmapEachPass)
                        PicPass = self.Pictures[batch][channel*self.r:(channel+1)*self.r][
                                     head:tail]
                        WeightPass = self.FilterWeights[ofmap*self.t:(ofmap+1)*self.t][
                                     channel*self.r:(channel+1)*self.r][head:tail]
                        Passes.append([PicPass, WeightPass])
                        head += self.e #or conf.EyerissWidth equivalently
        return Passes
    
    def __SetMappingParameters__(self, m=0, n=0, e=0, p=0, q=0, r=0, t=0):
        self.m = m if m!=0 else self.m
        self.n = n if n!=0 else self.n
        self.p = p if p!=0 else self.p
        self.q = q if q!=0 else self.q
        self.e = e if e!=0 else self.e
        self.r = r if r!=0 else self.r
        self.t = t if t!=0 else self.t
    
    def __PEArrayMapping__(self):
        #TODO: also consider stride
        PESetHeight = self.FilterWeights.shape[3] #filter height
        PESetWidth = self.Pictures.shape[3]- self.FilterWeights.shape[3] + 1 #ofmap height
        #Eyeriss only support filter height smaller than PE array height
        assert PESetHeight <= conf.EyerissHeight
        t = int(conf.EyerissHeight/PESetHeight) # t filters
        
        #TODO: let's assume PESetW >=PEArrayWidth for now
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
        m = self.FilterWeights.shape[0]/(self.r*self.q)
        self.__SetMappingParameters__(m=m, e=e,t=t)
        
    def __PESetMapping__(self):
        #TODO: add reusing filter, processing n ifmaps at a time
        slidingWindow = self.Pictures.shape[3]
        q = int(conf.IfmapSpad/(slidingWindow*self.n))
        p = min(int(conf.PsumSpad/self.n), int(conf.FilterSpad/(q*slidingWindow))
        #TODO:channel, filter number should be divisible by q, p
        self.__SetMappingParameters__(q=q,p=p)
        
        self.__FmapReuse__()
        self.__FilterReuse__()
        self.__ChannelAccumulation__()
        
        
    def __SetPicAndFlt__(self, Pictures=None, FilterWeights=None):
        self.Pictures = Pictures if Pictures !=None else self.Pictures
        self.FilterWeights = FilterWeights 
                      if FilterWeigths != None else self.FilterWeights
        
    def __FilterReuse__(self):
        if self.n > 1:
            assert self.Pictures.shape[0]%self.n == 0
            Pictures = np.split(self.Pictures, self.n)
            Pictures = np.concatenate(Pictures, axis = 3)
                
            self.__SetPicAndFlt__(Pictures=Pictures)

    def __FmapReuse__(self):
        if self.p > 1: 
            assert self.FilterWeights.shape[0]%self.p == 0
            s = np.array(self.FilterWeights.shape)
            s[0] /= self.p
            s[3] *= self.p
            FilterWeights = np.empty(s,dtype=self.FilterWeights.dtype)
            for p in range(self.p):
                FilterWeights[:,:,:, p::self.p] = self.FilterWeights[p::self.p]
            
            self.__SetPicAndFlt__(FilterWeights = FilterWeights)

    def __ChannelAccumulation__(self):
        if self.q > 1:
            assert self.FilterWeights.shape[1]%self.q == 0
            
            s = np.array(self.Pictures.shape)
            s[1] /= self.q
            s[3] *= self.q
            Pictures = np.empty(s,dtype=self.Pictures.dtype)
            for q in range(self.q):
                Pictures[:,:,:, q::self.q] = self.Pictures[:,q::self.q,:,:]
                
            s = np.array(FilterWeights.shape)
            s[1] /= self.q
            s[3] *= self.q
            FilterWeights = np.empty(s,dtype=self.FilterWeights.dtype)
            for q in range(self.q):
                FilterWeights[:,:,:, q::self.q] = self.FilterWeights[:,q::self.q,:,:]
        
            self.__SetPicAndFlt__(Picture, FilterWeights)
        
    def AccumulateChannel(self,Psum):
        for channel in range(int(self.Pictures.shape[1]/self.r)):
            Passes.append
    def Reverse(self, Psum):
        #TODO: let's ignore send back psum to PEs for now
        index = 0
        ofmapWidth = self.Pictures.shape[2] - self.FilterWeigths.shape[2] + 1
        OfMaps = np.zeros( (self.Picture.shape[0], self.FilterWeights.shape[0], 
                            ofmapWidth, ofmapWidth*self.n*self.p ))
        for batch in range( self.Pictures.shape[0] ):
            ofMap = []
            for ofmap in range( int(self.FilterWeights.shape[0]/self.t) ):
                SumRow = []
                for channel in range( int(self.Pictures.shape[1]/self.r) ): 
                    head = 0
                    PsumRow = []
                    for e in range( int(ofmapWidth/self.e) ):
                        PsumRow.append( np.array(Psum[i]) )
                        index += 1
                    PsumRow = np.concatenate(PsumRow, axis=1)
                    assert PsumRow.shape == (self.t,ofmapWidth, 
                           ofmapWidth*self.n*self.p)
                    SumRow.append(PsumRow)
                SumRow = np.array(SumRow).sum(axis=0)
                ofMap.append(SumRow)
            OfMaps[batch] = np.concatenate(ofMap)
        self.__SetOfMaps__(OfMaps)
        self.__ReverseFmapReuse__()
        self.__ReverseFilterReuse__()
        

    def __ReverseFmapReuse__(self):
        s = np.array(self.OfMaps.shape)
        s[1] *= self.p
        s[3] /= self.p
        OfMaps = np.zeros(s, dtype=self.OfMaps.dtype)
        for n in range(len(self.n)):
            OfMaps[n::self.n] = self.OfMaps[:,:,:,n::self.n]

        self.__SetOfMaps__(m)

    def __ReverseFilterReuse__(self):
        OfMaps = np.split(self.OfMaps, int(self.OfMaps.shape()[3] / self.n), axis=3)
        OfMaps = np.concatenate(OfMaps, axis = 0)
        self.__SetOfMaps__(OfMaps)

    def __SetOfMaps__(self, OfMaps):
        self.OfMaps = OfMaps

    def Output(self):
        #TODO: trace memory write
        return self.Compress(self.ReturnImgs)

    def FullConnect(self, v1, v2, activation=1):
        return np.array(np.dot(v1, v2) / activation, dtype=int)
