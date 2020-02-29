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
        #self.RLE = IO2.RLE(RateNeed = 0)
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
        
    def Conv2d(self, Pictures=0, FilterWeights=0):
        #Pictures = self.RLE.Compress(Pictures)
        #FilterWeights = self.RLE.Compress(FilterWeights)
        Passes = self.input(Pictures, FilterWeights)
        ofmapWidth = Pictures.shape[2] - FilterWeights.shape[2] + 1
        Psum = [self.EyerissF.Conv2d(ps, ofmapWidth, self.n, self.p, self.q) for ps in Passes]
        self.Reverse(Psum)
        return self.Output()
            
    def ReLU(self, array):
        return Activiation.ReLU(array)

    def Pooling(self, array):
        return Pooling.Pooling(array)
            
    def input(self, Pictures, FilterWeights):
        #Pictures = self.RLE.Decompress(Pictures)
        #FilterWeights = self.RLE.Decompress(FilterWeights)
        self.__SetPicAndFlt__(Pictures, FilterWeights)
        return self.Conv2DMapping()

        
    def Conv2DMapping(self):
        self.__PEArrayMapping__()
        #self.__PESetMapping__()
        return self.__SetPasses__()

    def __SetPasses__(self):
        Passes = []
        for batch in range( self.Pictures.shape[0] ):
            for ofmap in range( int(self.FilterWeights.shape[0]/self.t) ):
                for channel in range( int(self.Pictures.shape[1]/self.r) ):
                    #TODO: let's assume there is no reuse of filter vertically
                    #self.e can be smaller than actual ofmapWidth
                    #e.g. self.e of 27 is sent out, so 14 for one and 13 for another pass
                    #for outmapwidth of 14, filterwidth of 3, we need 14+3-1 ifmaps
                    ifmapEachPass = self.e + self.FilterWeights.shape[2] - 1
                    ofmapWidth = self.Pictures.shape[2] - self.FilterWeights.shape[2] + 1
                    head = 0
                    for e in range( int(ofmapWidth/self.e) ):
                        tail = head+ifmapEachPass
                        PicPass = self.Pictures[batch, channel*self.r:(channel+1)*self.r,
                                     head:tail, :]
                        WeightPass = self.FilterWeights[ofmap*self.t:(ofmap+1)*self.t, 
                                    channel*self.r:(channel+1)*self.r, :, :]
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
        PESetHeight = self.FilterWeights.shape[2] #filter height
        PESetWidth = self.Pictures.shape[2]- self.FilterWeights.shape[2] + 1 #ofmap height
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
            
        self.__SetMappingParameters__(e=e,t=t)
        
    def __PESetMapping__(self):
        #TODO: add reusing filter, processing n ifmaps at a time
        slidingWindow = self.FilterWeights.shape[2]
        qMax = int(conf.IfmapSpad/(slidingWindow*self.n))
        q=qMax
        for q in range(qMax,0,-1):
            if self.FilterWeights.shape[1]%q == 0:
                break
        pMax = min(int(conf.PsumSpad/self.n), int(conf.FilterSpad/
                (q*slidingWindow)))
        # sometimes we don't need large p at PE level 
        # since PE array already reused it
        pMax = min(pMax, int(self.FilterWeights.shape[0]/self.t))  
        p=pMax
        for p in range(pMax,0,-1):
            if self.FilterWeights.shape[0]%p == 0:
                break
        m = self.FilterWeights.shape[0]/(self.r*q)
        self.__SetMappingParameters__(q=q,p=p)
        
        self.__FilterReuse__()
        self.__FmapReuse__()
        self.__ChannelAccumulation__()
        
        
    def __SetPicAndFlt__(self, Pictures=None, FilterWeights=None):
        if isinstance(Pictures, (np.ndarray)): self.Pictures = Pictures
        if isinstance(FilterWeights, (np.ndarray)): self.FilterWeights = FilterWeights  
        
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
                
            s = np.array(self.FilterWeights.shape)
            s[1] /= self.q
            s[3] *= self.q
            FilterWeights = np.empty(s,dtype=self.FilterWeights.dtype)
            for q in range(self.q):
                FilterWeights[:,:,:, q::self.q] = self.FilterWeights[:,q::self.q,:,:]
        
            self.__SetPicAndFlt__(Pictures, FilterWeights)
        

    def Reverse(self, Psum):
        #Psum is a list of Psums in the shape of [self.t, self.e, ofmapwidth*p*n]
        index = 0
        ofmapWidth = self.Pictures.shape[2] - self.FilterWeights.shape[2] + 1
        OfMaps = np.zeros( (self.Pictures.shape[0], self.FilterWeights.shape[0], 
                            ofmapWidth, ofmapWidth*self.n*self.p ))
        for batch in range( self.Pictures.shape[0] ):
            ofMap = []
            for ofmap in range( int(self.FilterWeights.shape[0]/self.t) ):
                SumRow = []
                for channel in range( int(self.Pictures.shape[1]/self.r) ): 
                    head = 0
                    PsumRow = []
                    for e in range( int(ofmapWidth/self.e) ):
                        PsumRow.append( np.array(Psum[index]) )
                        index += 1
                    PsumRow = np.concatenate(PsumRow, axis=1)
                    assert PsumRow.shape == (self.t,ofmapWidth, 
                           ofmapWidth*self.n*self.p)
                    SumRow.append(PsumRow)
                #TODO: let's ignore send back psum to PEs for now
                SumRow = np.array(SumRow).sum(axis=0)
                ofMap.append(SumRow)
            OfMaps[batch] = np.concatenate(ofMap)
        self.__SetOfMaps__(OfMaps)
        #self.__ReverseFmapReuse__()
        #self.__ReverseFilterReuse__()
        

    def __ReverseFmapReuse__(self):
        s = np.array(self.OfMaps.shape)
        s[1] *= self.p
        s[3] /= self.p
        OfMaps = np.zeros(s, dtype=self.OfMaps.dtype)
        for p in range(self.p):
            OfMaps[:,p::self.p] = self.OfMaps[:,:,:,p::self.p]
        self.__SetOfMaps__(OfMaps)

    def __ReverseFilterReuse__(self):
        OfMaps = np.split(self.OfMaps, self.n, axis=3)
        OfMaps = np.concatenate(OfMaps, axis = 0)
        self.__SetOfMaps__(OfMaps)

    def __SetOfMaps__(self, OfMaps):
        self.OfMaps = OfMaps

    def Output(self):
        #TODO: trace memory write
        #return self.Compress(self.ReturnImgs)
        return self.OfMaps
    def FullConnect(self, v1, v2, activation=1):
        return np.array(np.dot(v2, v1) / activation, dtype=int)
