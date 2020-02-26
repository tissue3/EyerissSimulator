import numpy as np
import conf


class PE:

    def __init__(self):
        self.IfmapSpad=conf.IfmapSpad
        self.FilterSpad=conf.FilterSpad
        self.PsumSpad=conf.PsumSpad
        self.PEState = conf.ClockGate
    
    def SetPEState(self, State):
        self.PEState = State

    def SetFilterRow(self, FilterRow):
        self.FilterRow = FilterRow

    def SetImageRow(self, ImageRow):
        self.ImageRow = ImageRow

    def SetChannelNum(self, ChannelNum):
        self.ChannelNum = ChannelNum
    def SetFilterNum(self, FilterNum):
        self.FilterNum = FilterNum
    def SetImageNum(self, ImageNum):
        self.ImageNum = ImageNum

    def __SetPsum__(self, Psum):
        self.Psum = Psum

    def __Conv1d__(self, ImageRow, FilterRow):
        PsumRow = list()
        for x in range(0, len(ImageRow) ):
            y = x + len(FilterRow)
            r = ImageRow[x:y] * FilterRow
            PsumRow.append(r.sum())
        return np.array(PsumRow)
        return np.add.reduceat(PsumRow, np.arange(0, len(PsumRow), self.Channel))

    def __Conv__(self):
        Image1 = self.ImageRow.reshape(-1, self.ChannelNum).T
        Filter1 = self.FilterRow.reshape(-1, self.ChannelNum).T
        psum1 = []
        for c in range(self.ChannelNum):
            Image2 = Image1[c]
            Filter2 = Filter1[c].reshape(-1, self.ImageNum).T
            psum2 = []
            for i in range(self.ImageNum):
                Image3 = np.array(np.split(Image2, self.FilterNum))
                Filter3 = Filter2[i]
                psum3 = []
                for f in range(self.FilterNum):
                    psum = self.__Conv1d__(Image3[f], Filter3)
                    psum3.append(psum)
                psum3 = np.concatenate(psum3)
                psum2.append(psum3)
            psum2 = np.array(psum2).T.reshape(-1)
            psum1.append(psum2)
         Psum = np.array(psum1).sum(axis=0)
         return Psum
         
    def CountPsum(self):
    #TODO: communicating PE with each other
        if self.PEState == conf.ClockGate:
            self.__SetPsum__(0)
        elif self.PEState == conf.Running:
            self.__SetPsum__(self.__Conv__())
