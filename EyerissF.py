import numpy as np
import conf
from PE import PE

class EyerissF:

    def __init__(self):
        self.PEArrayHeight = conf.EyerissHeight
        self.PEArrayWidth = conf.EyerissWidth
        self.__InitPEs__()

    def Conv2d(self, Pass, n, p, q):
        Picture, FilterWeight = Pass
        PESetH, PESetW = self.__DataDeliver__(Pictures, FilterWeights, q)
        self.__ShowStates__()
        self.__run__()
        Psums = self.__PsumTransport__(PESetH, PESetW, FilterWeights.shape[0], 
                Pictures.shape[0], p, n)
        self.__SetALLPEsState__(conf.ClockGate)

        return ReluedConvedArray

    def __InitPEs__(self):
    
        self.PEArray = list()
        for x in range(0, self.PEArrayHeight):
            self.PEArray.append(list())
            for y in range(0, self.PEArrayWidth):
                self.PEArray[x].append(PE())

    def __SetALLPEsState__(self, State):
    
        for ColumnELement in range(0, EyerissF.EyerissHeight):
            for RowElement in range(0, EyerissF.EyerissWidth):
                self.PEArray[ColumnELement][RowElement].SetPEState(State)


    def __DataDeliver__(self, Pictures, FilterWeights, n, p, q):
        # put the pic and filter row data into PEArray
        #TODO: change this function to network.DataDeliever()
        PESetH = FilterWeights.shape[2]
        PESetW = Pictures.shape[1] - FilterWeights.shape[2] + 1
        #TODO: let's assume PESetW >=PEArrayWidth for now
        # because current information is not enough to describe 2D mapping
        # we might introduce more parameters (tx, ty, rx, rz) in the future.
        y = -1
        for f in range(FilterWeights.shape[0]):#filters
            for c in range(Pictures.shape[0]):#channel
                for h in range(PESetH):
                    y += 1
                    for w in range(PESetW):
                        x = w % self.PEArrayWidth
                        self.PEArray[y][x].SetFilterRow(FilterWeights[f][c][h])
                        self.PEArray[y][x].SetImageRow(Pictures[c][h+w])
                        self.PEArray[y][x].SetChannelNum(q)
                        self.PEArray[y][x].SetFilterNum(p)
                        self.PEArray[y][x].SetImageNum(n)
                        self.PEArray[y][x].SetPEState(conf.ConvState)
                    

    def __run__(self):
        for y in range(0, conf.EyerissHeight):
            for x in range(0, conf.EyerissWidth):
                if self.PEArray[y][x].PEState != conf.ClockGate:
                    self.PEArray[y][x].CountPsum()

    def __PsumTransport__(self, PESetH, PESetW, ChannelNum, FilterNum, p, n):
        Psums = np.zeros((FilterNum, PESetH, PESetW,p*n*PESetW))
        for f in range(FilterNum):
            for c in range(ChannelNum):
                for h in range(PESetH):
                    for w in range(PESetW):
                        y = (f*FilterNum+c)*ChannelNum+h
                        x = w % self.PEArrayWidth
                        Psum = 0
                        if c!= ChannelNum-1:
                            Psum = self.PEArray[y][x].getPSum()
                            assert len(Psum) == PESetW*p*n
                            Psums[f][h][w] = Psum
                        y = (f*FilterNum+c+1)*ChannelNum+h
                        if c!= 0:
                            assert Psum != 0
                            self.PEArray[y][x].setPSumRow(PSum)
                            self.PEArray[y][x].CountPsum()
                        self.PEArray[y][x].SetPEState(conf.SumState)
        return Psums

    def __ShowPEState__(self, x, y):
        print("PE is : ", x, ",", y)

        if self.PEArray[x][y].PEState == conf.Running:
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
                if self.PEArray[x][y].PEState == conf.Running:
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

                if self.PEArray[x][y].PEState == conf.Running:
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

                if self.PEArray[x][y].PEState == conf.Running:
                    c = c + 1
                    yy.append(1)
                else:
                    yy.append(0)
            xx.append(yy)
            yy = []
        print("There are", c, "running PEs.")
        print(np.array(xx))
