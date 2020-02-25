import numpy as np


class RLE():
    
    def __init__(self, RateNeed = 0):
        self.RateNeed = RateNeed
        
    def Compress2D(NpArray):
        if NpArray.ndim == 1:
            Row = NpArray.shape
            Column = 1
        #elif (NpArray.shape) == 2:
        Row, Column = NpArray.shape
        #else:
        ComedNpArray = np.array([Row, Column])
        # NpArray=NpArray.flatten()
        NpArray = NpArray.reshape((1, Row * Column))
        ZeroCounter = 0
        for iterr in range(NpArray.size):
            if NpArray[0][iterr] == 0:
                ZeroCounter = ZeroCounter + 1
            #TODO: check zerocounter is not larger than 31; aslo change decompress.
            else:
                if ZeroCounter == 0:
                    ComedNpArray = np.append(ComedNpArray, np.array(NpArray[0, iterr]))
                else:
                    ComedNpArray = np.append(ComedNpArray, np.array([0, ZeroCounter, NpArray[0, iterr]]))
                    ZeroCounter = 0

        if ZeroCounter != 0:
            ComedNpArray = np.append(ComedNpArray, np.array([0, ZeroCounter]))
        if self.RateNeed == 0:
            return ComedNpArray
        else:
            #before run length encoding, 64b can store 4 numbers
            #after RLE, 64b can store 3 non-zero nubmers and zeros among them
            CompressRate = float(ComedNpArray.size)/3 / (Row * Column)/4
            print("CompressRate is :",CompressRate)
            return ComedNpArray

    def Decompress2D(NpArray):
        Length = NpArray.size - 2
        Row = NpArray[0]
        Column = NpArray[1]
        DecomedNpArray = list()
        for interr in range(Length):

            if NpArray[interr + 2] == 0:
                for x in range(NpArray[interr + 3]):
                    DecomedNpArray.append(0)
            elif NpArray[interr + 1] == 0:  # NpArray[interr+2] != 0
                pass

            else:  # NpArray[interr+2] != 0 && NpArray[interr+1] != 0
                DecomedNpArray.append(NpArray[interr + 2])
        DecomedNpArray = np.array(DecomedNpArray, dtype=int)
        DecomedNpArray = DecomedNpArray.reshape(Row, Column)
        return DecomedNpArray

    def Compress(array):
        assert len(array.shape) >= 2
        if len(array.shape) == 2:
            return self.Compress2D(x)
        return np.array([self.Compress(x) for x in array])

    def Decompress(array):
        assert len(array.shape) >= 2
        if len(array.shape) == 2:
            return self.Decompress2D(x)
        return [Decompress(x) for x in array]
