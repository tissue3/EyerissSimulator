import numpy as np
from . import conf
from .PE import PE

class EyerissF:

    def __init__(self):
        self.PEArrayHeight = conf.EyerissHeight
        self.PEArrayWidth = conf.EyerissWidth
        self.__InitPEs__()

    def Conv2d(self, Pass, OfMapWidth, n, p, q, showStates=0):
        Pictures, FilterWeights = Pass
        PESetH, PESetW, FilterNum, ChannelNum = self.__DataDeliver__(
                        Pictures, FilterWeights, n, p, q)
        if showStates: self.__ShowStates__()
        self.__run__()
        self.__PsumTransportLN__(PESetH, PESetW, FilterNum, ChannelNum, n, p)
        self.__PsumTransportGIN__(PESetH, PESetW, FilterNum, ChannelNum, n, p)
        Psums = self.__DataCollect__(PESetH, PESetW, FilterNum, ChannelNum, OfMapWidth, n, p)
        self.__SetALLPEsState__(conf.ClockGate)

        return Psums

    def __InitPEs__(self):
        self.PEArray = list()
        for x in range(0, self.PEArrayHeight):
            self.PEArray.append(list())
            for y in range(0, self.PEArrayWidth):
                self.PEArray[x].append(PE())

    def __SetALLPEsState__(self, State):
        for ColumnELement in range(0, conf.EyerissHeight):
            for RowElement in range(0, conf.EyerissWidth):
                self.PEArray[ColumnELement][RowElement].SetPEState(State)


    def __DataDeliver__(self, Pictures, FilterWeights, n, p, q):
        # put the pic and filter row data into PEArray
        #TODO: change this function to network.DataDeliever()
        PESetH = FilterWeights.shape[2]
        PESetW = Pictures.shape[1] - FilterWeights.shape[2] + 1
        #TODO: let's assume PESetW >=PEArrayWidth for now
        # because current information is not enough to describe 2D mapping
        # we might introduce more parameters (tx, ty, rx, rz) in the future.
        FilterNum = FilterWeights.shape[0]
        ChannelNum = Pictures.shape[0]
        for f in range(FilterNum):#filters
            for c in range(ChannelNum):#channel
                for h in range(PESetH):
                    for w in range(PESetW):
                        x = w % self.PEArrayWidth
                        offsetY = int(w/self.PEArrayWidth)*PESetH*ChannelNum*FilterNum
                        y = offsetY+(f*ChannelNum+c)*PESetH+h
                        self.PEArray[y][x].SetFilterRow(FilterWeights[f][c][h])
                        self.PEArray[y][x].SetImageRow(Pictures[c][h+w])
                        self.PEArray[y][x].SetChannelNum(q)
                        self.PEArray[y][x].SetFilterNum(p)
                        self.PEArray[y][x].SetImageNum(n)
                        self.PEArray[y][x].SetPEState(conf.ConvState)
        return PESetH, PESetW, FilterNum, ChannelNum
                  
    def __DataCollect__(self, PESetH, PESetW, FilterNum, ChannelNum, OfMapWidth, n, p):
        Psums = np.zeros( (FilterNum, PESetW, OfMapWidth*n*p) )
        for f in range(FilterNum):#filters
            for w in range(PESetW):
                x = w % self.PEArrayWidth
                offsetY = int(w/self.PEArrayWidth)*PESetH*ChannelNum*FilterNum
                y = offsetY+(f*ChannelNum+ChannelNum-1)*PESetH+PESetH-1
                Psums[f][w] = self.PEArray[y][x].getPsumRow()
        return Psums         
        
    def __run__(self):
        for y in range(self.PEArrayHeight):
            for x in range(self.PEArrayWidth):
                if self.PEArray[y][x].PEState != conf.ClockGate:
                    self.PEArray[y][x].CountPsum()

    def __PsumTransportLN__(self, PESetH, PESetW, FilterNum, ChannelNum, n, p):
        for f in range(FilterNum):
            for c in range(ChannelNum):
                for w in range(PESetW):
                    for h in range(PESetH - 1):
                        offsetY = int(w/self.PEArrayWidth)*PESetH*ChannelNum*FilterNum
                        y = offsetY+(f*ChannelNum+c)*PESetH+h
                        x = w % self.PEArrayWidth
                        Psum = self.PEArray[y][x].getPsumRow()
                        assert len(Psum) == PESetW*p*n
                        y += 1
                        self.PEArray[y][x].SetInPsumRow(Psum)
                        self.PEArray[y][x].SetPEState(conf.SumState)
                        self.PEArray[y][x].CountPsum()
        
    def __PsumTransportGIN__(self, PESetH, PESetW, FilterNum, ChannelNum, n, p):
        for f in range(FilterNum):
            for c in range(ChannelNum):
                for w in range(PESetW):
                    offsetY = int(w/self.PEArrayWidth)*PESetH*ChannelNum*FilterNum
                    y = offsetY+(f*ChannelNum+c)*PESetH+PESetH-1
                    x = w % self.PEArrayWidth
                    Psum = self.PEArray[y][x].getPsumRow()
                    assert len(Psum) == PESetW*p*n
                    if c!=ChannelNum-1:
                    #transport to another channel
                        y = offsetY+(f*ChannelNum+c+1)*PESetH+PESetH-1
                        self.PEArray[y][x].SetInPsumRow(Psum)
                        self.PEArray[y][x].SetPEState(conf.SumState)
                        self.PEArray[y][x].CountPsum()

    def __ShowPEState__(self, x, y):
        print("PE is : ", x, ",", y)

        if self.PEArray[x][y].PEState != conf.ClockGate:
            print("PEState : Running")

        else:
            print("PEState : ClockGate")

        print("FilterWeight :", self.PEArray[x][y].FilterWeight)
        print("ImageRow :", self.PEArray[x][y].ImageRow)

    def __ShowAllPEState__(self):

        xx = list()
        yy = list()
        for x in range(conf.EyerissHeight):
            for y in range(conf.EyerissWidth):
                self.__ShowPEState__(x, y)
                if self.PEArray[x][y].PEState != conf.ClockGate:
                    yy.append(1)
                else:
                    yy.append(0)

            xx.append(yy)
            yy = []
        print(np.array(xx))

    def __ShowRunningPEState__(self):

        c = 0
        xx = list()
        yy = list()
        for x in range(conf.EyerissHeight):
            for y in range(conf.EyerissWidth):

                if self.PEArray[x][y].PEState != conf.ClockGate:
                    self.__ShowPEState__(x, y)
                    c = c + 1
                    yy.append(1)
                else:
                    yy.append(0)
            xx.append(yy)
            yy = []
        print("There are", c, "running PEs.")
        print(np.array(xx))

    def __ShowStates__(self):
        c = 0
        xx = list()
        yy = list()
        for x in range(conf.EyerissHeight):
            for y in range(conf.EyerissWidth):

                if self.PEArray[x][y].PEState != conf.ClockGate:
                    c = c + 1
                    yy.append(1)
                else:
                    yy.append(0)
            xx.append(yy)
            yy = []
        print("There are", c, "running PEs.")
        print(np.array(xx))
