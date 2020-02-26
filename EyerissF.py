import numpy as np
import conf
from PE import PE
from Activiation import Relu

class EyerissF:

    def __init__(self):
        self.PEArrayHeight = conf.EyerissHeight
        self.PEArrayWidth = conf.EyerissWidth
        self.__InitPEs__()

    def Conv2d(self, Pass, n, p, q):
        Picture, FilterWeight = Pass
        PictureColumnLength, FilterWeightColumnLength = self.__DataDeliver__(Pictures, FilterWeights, q)
        self.__ShowStates__()
        self.__run__()
        ConvedArray = self.__PsumTransport__(PictureColumnLength, FilterWeightColumnLength)
        ReluedConvedArray = Relu(ConvedArray)
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
        #TODO: change this function to network.DataDeliever()
        # put the pic and filter row data into PEArray
        PESetH = FilterWeights.shape[2]
        PESetW = Pictures.shape[1] - FilterWeights.shape[2] + 1
        
        #pic
        y = 0
        for f in range(self.FilterWeights.shape[0]): #filters
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
                        self.PEArray[y][x].SetPEState(conf.Running)
                    

    def __run__(self):
        for x in range(0, conf.EyerissHeight):
            for y in range(0, conf.EyerissWidth):
                if self.PEArray[x][y].PEState == conf.Running:
                    self.PEArray[x][y].CountPsum()

    def __PsumTransport__(self, PictureColumnLength, FilterWeightColumnLength):
        #send psum to PE if multiple ifmap channel (Pictures.shape[0]) > 1
        #PE do the addition
        
        line = list()
        result = list()
        for RowElement in range(0, PictureColumnLength + 1 - FilterWeightColumnLength):

            # 清空list
            line.clear()
            for ColumnElement in range(0, FilterWeightColumnLength).__reversed__():
                # 从上到下把psum加入list
                line.append(self.PEArray[ColumnElement][RowElement].Psum)

            # 将list中的Psum做和，得到一行卷积值，保存到r中
            result.append(np.sum(line, axis=0, dtype=int)) 
        # 将r中全部的卷积值组合成一个矩阵，并返回
        if result == []:
            return

        return np.vstack(result)

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
        print("一共有", c, "个PE正在运行")
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
