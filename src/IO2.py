import numpy as np


class RLE():
    
    def __init__(self, RateNeed = 0):
        self.RateNeed = RateNeed
        
    def __Compress1D__(self, NpArray):
        assert NpArray.ndim == 1
        Length = NpArray.size
        ComedNpArray = []
        ZeroCounter = 0
        for iterr in range(Length):
            if NpArray[iterr] == 0:
                ZeroCounter = ZeroCounter + 1
                if ZeroCounter == 32:
                    ComedNpArray = ComedNpArray+[31, 0]
                    ZeroCounter -= 32
            else:
                ComedNpArray = ComedNpArray+[ZeroCounter, NpArray[iterr]]
                ZeroCounter = 0
        if ZeroCounter > 0:
            ComedNpArray = ComedNpArray+[ZeroCounter-1, 0]
        return ComedNpArray

    def __Decompress1D__(self, Array):
        assert not isinstance(Array[0], list)
        Length = len(Array)
        assert Length % 2 == 0
        DecomedNpArray = np.zeros(Array[0])
        DecomedNpArray = np.append(DecomedNpArray, Array[1])
        for iterr in range(2, Length, 2):
            DecomedNpArray = np.append(DecomedNpArray, np.zeros(Array[iterr]))
            DecomedNpArray = np.append(DecomedNpArray, Array[iterr+1])
        return DecomedNpArray
        
    def __CompressND__(self, Array):
        if len(Array.shape) == 1:
            ComedArray = self.__Compress1D__(Array)
            Length = len(ComedArray)
            assert Length % 2 == 0
            #number of run the compressed array represents
            RunNum = ( int( (Length/2 - 1)/3 + 1)*3)
            return ComedArray, RunNum
        RunNum = 0
        ComedArray = []
        for Arr in Array:
            ComedArr, Run = self.__CompressND__(Arr)
            RunNum += Run
            ComedArray.append(ComedArr)
        return ComedArray, RunNum
    
    def Compress(self, Array):
        ComedArray, RunNum = self.__CompressND__(Array)
        if self.RateNeed:
            print("The compression ratio is", (Array.size/4) / (RunNum/3))
        return ComedArray
    
    def Decompress(self, Array):
        if not isinstance(Array[0], list):
            return self.__Decompress1D__(Array)
        return np.array([self.Decompress(array) for array in Array])
        
        
