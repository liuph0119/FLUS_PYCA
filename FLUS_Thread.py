#coding:utf-8
from PyQt5.QtCore import *
import FLUS_Utils
import numpy as np
import time, os
import xml.etree.cElementTree as ET
from sklearn.externals import joblib
from gdalconst import *


class NNTrainingThread(QThread):
    """
    采样 + NN 训练 + 预测的外部线程，与界面主线程不同，可以引入防止界面假死
    """
    # 结束信号
    finished = pyqtSignal()
    # 发送信号，传给主界面，用于显示在状态栏
    sendMsg = pyqtSignal(str)
    # 发送信号，传给主界面，用于弹出对话框
    sendMsg2Box = pyqtSignal(str, str)

    def __init__(self, parent):
        QThread.__init__(self, parent)

    def setParam(self, wd, xmlfn):
        """
        为线程设置参数
        :param wd: 工作路径，用于查找xml文件，写入发展概率文件
        :param xmlfn: xml文件名
        :return:
        """
        self.sWorkDirectory = wd
        self.xmlfn = xmlfn

    def parseXML(self):
        """
        解析xml
        :return: None if fail, 1 if success
        """
        xmlfn = self.sWorkDirectory + "/" + self.xmlfn
        if (not os.path.exists(xmlfn)):
            # print("ann_config.xml does not exist")
            self.sendMsg2Box.emit("Please set ANN-based Probability\nEstimation Parameters First!", "error")
            return None


        else:
            tree = ET.parse(xmlfn)
            root = tree.getroot()
            # = = = = 解析 xml 文件 = = = = = = = = = = =
            self.sLUFileName = root.find("FilePaths").find("LandUseFilePath").text  # 土地利用数据文件名
            self.sPgFileName = root.find("FilePaths").find("ProbabilityFilePath").text  # 存储的发展概率文件名
            self.vsFeatFileName = []
            for feat in root.find("FilePaths").find("FeatureFilePaths").findall("FeatureFilePath"):
                self.vsFeatFileName.append(feat.text)  # 空间变量文件名，由于存在多张图像，因此存入列表

            # 从xml文件读取采样方法和采样比例
            self.sSampleMethod = root.find("Sampling").find("Method").text
            self.dSampleRate = float(root.find("Sampling").find("Ratio").text)

            # 从xml文件读取神经网络训练参数
            self.nHhiddenlayersize = int(root.find("NeuralNetwork").find("HiddenLayer").text)
            self.nMaxIter = int(root.find("NeuralNetwork").find("MaximumIteration").text)
            self.dAlpha = float(root.find("NeuralNetwork").find("Alpha").text)
            self.dLearningRate = float(root.find("NeuralNetwork").find("LearningRate").text)

            return 1

    def sampling(self):
        """
        采样
        :return:
        """
        self.poLU, msg = FLUS_Utils.loadImage(self.sLUFileName)
        self.dLUNodata = self.poLU.mdInvalidValue

        # 读取空间变量数据集，并且提出 nodata 值
        self.pppFeats = np.zeros((len(self.vsFeatFileName), self.poLU.mnRows, self.poLU.mnCols))
        self.vdFeatNodata = []
        for i in range(len(self.vsFeatFileName)):
            flusimg, msg = FLUS_Utils.loadImage(self.vsFeatFileName[i])
            self.vdFeatNodata.append(flusimg.mdInvalidValue)
            self.pppFeats[i] = flusimg.mpArray[0]

        # 采集样本
        self.sendMsg.emit("Sampling... [ %s / %.3f ]" % (self.sSampleMethod, self.dSampleRate))
        if (self.sSampleMethod == "Uniform"):
            self.ppSamples = FLUS_Utils.uniformSample2table(self.pppFeats, self.poLU.mpArray, self.dSampleRate, self.dLUNodata , self.vdFeatNodata)
        else:
            self.ppSamples = FLUS_Utils.randomSample2table(self.pppFeats, self.poLU.mpArray, self.dSampleRate, self.dLUNodata, self.vdFeatNodata)
        self.sendMsg.emit("Sample Success! Valid Sample Number=%d" % (self.ppSamples.shape[0]))

        # 将样本存入文件，文件名形如training_sample_20171220121212.csv
        self.sendMsg.emit("Save Sample File...")
        timestr = time.strftime('%Y%m%d%H%M%S')

        if(not os.path.exists(self.sWorkDirectory+"/training_samples")):
            os.mkdir(self.sWorkDirectory+"/training_samples")
        np.savetxt(self.sWorkDirectory + "/training_samples/training_sample_" + timestr + ".csv", self.ppSamples, delimiter=',')
        self.sendMsg.emit("Save Sample File Success!")

    def trainNN(self):
        """
        训练神经网络模型
        :return:
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        # 提取X, Y，用于训练
        X = self.ppSamples[:, :-1]
        Y = self.ppSamples[:, -1]
        del self.ppSamples

        # 构造神经网络，训练
        self.clf = MLPClassifier(hidden_layer_sizes=(self.nHhiddenlayersize,), max_iter=self.nMaxIter, alpha=self.dAlpha,
                       solver='sgd', tol=1e-9,
                      learning_rate_init=self.dLearningRate)
        # self.clf = RandomForestClassifier(n_estimators= 10)

        self.sendMsg.emit("Training...")
        self.clf.fit(X, Y)
        score = self.clf.score(X, Y)
        del X, Y

        joblib.dump(self.clf, self.sWorkDirectory + "/ANNModel.pkl")
        #self.sendMsg.emit("Training Success! Score = %.2f%%"%(score*100))
        self.sendMsg2Box.emit("Training Success! Score = %.2f%%"%(score*100), "info")

    def predictPg(self):
        """
        预测发展概率
        :return:
        """
        self.sendMsg.emit("Predicting...")
        # 首先获取类别数目
        validVals = np.unique(self.poLU.mpArray)
        classNum = validVals.shape[0]
        if (self.dLUNodata in validVals):
            classNum = validVals.shape[0] - 1

        probability = np.zeros((classNum, self.pppFeats.shape[1], self.pppFeats.shape[2]))

        # 这里可以开多线程
        for i in range(self.pppFeats.shape[1]):
            for j in range(self.pppFeats.shape[2]):
                arr = np.zeros((1, self.pppFeats.shape[0]))

                # 如果数据为 nodata，则跳过，发展概率赋值为 nodata
                nodataflag = abs(self.poLU.mpArray[0, i, j] - self.dLUNodata) > 1e-6
                for k in range(self.pppFeats.shape[0]):
                    arr[0, k] = self.pppFeats[k, i, j]
                    nodataflag = nodataflag and abs(arr[0, k] - self.vdFeatNodata[k]) > 1e-6

                if (nodataflag == False):
                    for k in range(classNum):
                        probability[k, i, j] = self.vdFeatNodata[0]
                    continue

                # 预测各类概率并且存入矩阵
                prob = self.clf.predict_proba(arr)
                for k in range(classNum):
                    probability[k, i, j] = prob[0, k]
            #self.progressBar.setValue(i / self.pppFeats.shape[1])
            if((i+1)%10 == 0 or (i+1)==self.pppFeats.shape[0]):
                self.sendMsg.emit("Predicting...%d/%d"%(i+1, self.pppFeats.shape[1]))

        # 保存文件
        self.sendMsg.emit("Saving Probability File...")
        if (FLUS_Utils.array2Tif(probability, self.sPgFileName, GDT_Float32, self.vdFeatNodata[0], self.poLU.mpoDataset.GetGeoTransform(),
                                 self.poLU.mpoDataset.GetProjection()) == None):
            self.sendMsg2Box.emit("File Exists!\nPlease Delete it first!", "error")
            self.finished.emit()

        # 清理内存
        del self.poLU, self.pppFeats, probability

    # 重构run函数
    def run(self):
        #self.finished.emit()
        if(self.parseXML()!= None):
            self.sampling()
            self.trainNN()
            self.predictPg()
        self.finished.emit()    # 这里千万不要写成finished().emit()!!!! debug了很久
                                # 动态类型一时爽，代码重构火葬场。

    def __del__(self):
        self.clf = None



class CAThread(QThread):
    """
    CA 线程
    """
    finished = pyqtSignal()
    sendMsg = pyqtSignal(str)
    sendMsg2Box = pyqtSignal(str, str)


    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.sWorkDirectory = None
        self.xmlfn = None
        self.nRows = 1                                                                      # 行数
        self.nCols = 1                                                                      # 列数
        self.nMaxIter = 300                                                                 # 最大迭代次数
        self.nWinSize = 7                                                                   # 邻域窗口大小
        self.dAccelerate = 0.1                                                              # 加速率
        self.nLUClassNum = 2                                                                # 土地利用类别数目
        self.pCurrentAmount = np.zeros((self.nLUClassNum))                                  # 当前各类用地像元数目，1d array
        self.pTargetAmount = np.zeros((self.nLUClassNum))                                   # 目标各类用地像元数目, 1d array
        self.ppCostMatrix = np.ones((self.nLUClassNum, self.nLUClassNum))                   # 类别之间的转换矩阵, 1d array
        self.pNeighborWeight = np.zeros((self.nLUClassNum))                                 # 各类用地的邻域权重, 1d array
        self.pppProbability = np.zeros((self.nLUClassNum, self.nRows, self.nCols))          # 神经网络训练的发展适宜性, 3d array
        self.ppRestrictive = np.zeros((self.nRows, self.nCols)   )                             # 限制性因素

        self.ppCurrentLU = np.zeros((self.nRows, self.nCols))                                # 当前用地分布， 2d array
        self.ppSimulateLU = np.zeros((self.nRows, self.nCols))                               # 模拟用地分布， 2d array


    def setParam(self, wd, xmlfn):
        """
        为线程设置参数
        :param wd: 工作路径，用于查找xml文件，写入发展概率文件
        :param xmlfn: xml文件名
        :return:
        """
        self.sWorkDirectory = wd
        self.xmlfn = xmlfn


    def parseXML(self):
        """
        解析参数
        :return:
        """
        xmlfn = self.sWorkDirectory + "/" + self.xmlfn
        if (not os.path.exists(xmlfn)):
            # print("ann_config.xml does not exist")
            self.sendMsg2Box.emit("Please set ANN-based Probability\nEstimation Parameters First!", "error")
            return None

        else:
            tree = ET.parse(xmlfn)
            root = tree.getroot()
            # = = = = 解析 xml 文件 = = = = = = = = = = =
            self.sCurLUFileName = root.find("FilePaths").find("InitialLandUseFilePath").text
            self.sSimLUFileName = root.find("FilePaths").find("SimulationLandUseFilePath").text
            self.sDevelopProbFileName = root.find("FilePaths").find("ProbabilityFilePath").text
            self.sRestrictiveFileName = root.find("FilePaths").find("RestrictiveFilePath").text


            self.nWinSize = int(root.find("SimulationParameters").find("NeighborSize").text)
            if(self.nWinSize % 2 == 0):
                self.nWinSize += 1
            self.nMaxIter = int(root.find("SimulationParameters").find("MaxIterationNum").text)
            self.dAccelerate = float(root.find("SimulationParameters").find("Accelerate").text)
            self.nLUClassNum = int(root.find("SimulationParameters").find("ClassNum").text)

            # 惯性系数
            self.pInertia = np.ones((self.nLUClassNum))

            # 读取邻域权重向量
            self.pNeighborWeight = np.zeros((self.nLUClassNum))
            weightNodes = root.find("SimulationParameters").find("NeighborWeights").findall("ClassWeight")
            for i in range(len(weightNodes)):
                self.pNeighborWeight[i] = float(weightNodes[i].text)

            # 获取当前各类像元数目
            self.pCurrentAmount = np.zeros((self.nLUClassNum))
            currentNodes = root.find("SimulationParameters").find("CurrentAmounts").findall("ClassAmount")
            for i in range(len(currentNodes)):
                self.pCurrentAmount[i] = int((currentNodes[i].text))


            # 读取目标像元数目
            self.pTargetAmount = np.zeros((self.nLUClassNum))
            targetNodes = root.find("SimulationParameters").find("TargetAmounts").findall("ClassAmount")
            for i in range(len(targetNodes)):
                self.pTargetAmount[i] = int((targetNodes[i].text))

            # 发展需求量
            self.pDemandAmount = self.pTargetAmount - self.pCurrentAmount
            # 读取类型间转换矩阵
            self.ppCostMatrix = np.zeros((self.nLUClassNum, self.nLUClassNum))
            costNodes = root.find("SimulationParameters").find("CostMatrix").findall("CostRow")
            for i in range(len(targetNodes)):
                costlist = (costNodes[i].text).split(",")
                for j in range(len(costlist)):
                    self.ppCostMatrix[i,j] = float(costlist[j])

            self.sendMsg.emit("parse xml success!")



    def init(self):
        """
        读取文件并申请内存：包括初始土地利用数据和模拟土地利用数据，发展概率，限制性因素
        :return:
        """
        # 读取影像
        luimg, msg = FLUS_Utils.loadImage(self.sCurLUFileName)

        self.nRows = luimg.mnRows
        self.nCols = luimg.mnCols
        self.pGeoTransform = luimg.mpoDataset.GetGeoTransform()
        self.sProjectionRef = luimg.mpoDataset.GetProjection()
        self.ppCurrentLU = luimg.mpArray[0]
        self.ppSimulateLU = luimg.mpArray[0]
        self.dLUNodata = luimg.mdInvalidValue
        self.gDataType = luimg.mgDataType
        del luimg

        probimg,msg = FLUS_Utils.loadImage(self.sDevelopProbFileName)
        self.pppProbability = probimg.mpArray
        self.dProbNodata = probimg.mdInvalidValue
        del probimg

        restrictiveimg, msg = FLUS_Utils.loadImage(self.sRestrictiveFileName)
        self.ppRestrictive = restrictiveimg.mpArray[0]
        self.dRestrictiveNodata = restrictiveimg.mdInvalidValue
        del restrictiveimg

        self.sendMsg2Box.emit("init success!", "info")



    def calNeighbor(self, r, c):
        """
        计算(r, c)位置处的邻域效应，
        :param r: 行
        :param c: 列
        :return:返回一个包含n个float数值的1d numpy array，其中n为用地类数
        """
        # get half size
        nHalfSize = (self.nWinSize - 1) / 2

        nLeft = int(c - nHalfSize)
        nRight = int(c + nHalfSize)
        nBottom = int(r + nHalfSize)
        nTop = int(r - nHalfSize)
        if(not c > nHalfSize):
            nLeft = 0
        if(not c < self.nCols-nHalfSize):
            nRight = self.nCols-1
        if(not r > nHalfSize):
            nTop = 0
        if(not r < self.nRows-nHalfSize):
            nBottom = self.nRows - 1

        pNeighborEffect = np.zeros(self.nLUClassNum)
        dNeighborTotal = 0.0
        for i in range(nTop, nBottom+1):
            for j in range(nLeft, nRight+1):
                val = int(self.ppCurrentLU[i,j])

                if (val > self.nLUClassNum or val < 1 or i==j):
                    continue
                pNeighborEffect[val-1] += self.pNeighborWeight[val-1]*1
                dNeighborTotal += self.pNeighborWeight[val-1]

        if(dNeighborTotal > 1):
            pNeighborEffect /= (dNeighborTotal)
        else:
            pNeighborEffect = np.zeros(self.nLUClassNum)
        return pNeighborEffect


    def iteration(self):

        nIter = 0
        bFlag = False
        while not bFlag and nIter < self.nMaxIter:
            # bFlag = True
            # for current, target in zip(self.pCurrentAmount, self.pTargetAmount):
            #     bFlag = bFlag and (current < target)
            # # 各类像元数目均达标，则满足要求
            # if(not bFlag):
            #     break

            print ("iter %d  current demand: "%(nIter+1), self.pDemandAmount, end = '')
            self.sendMsg.emit("iteration %d/%d..."%(nIter+1, self.nMaxIter))
            samplenum = 0
            # 每次随机选取1000个点
            while samplenum < self.nRows*self.nCols*0.01:
                r = np.random.randint(0, self.nRows)
                c = np.random.randint(0, self.nCols)

                val = int(self.ppSimulateLU[r,c])
                pc = self.ppRestrictive[r, c]  # 限制因素

                # 如果是nodata
                if(abs(val - self.dLUNodata) < 1e-6 or abs(self.pppProbability[0,r,c] - self.dProbNodata) < 1e-6 or abs(pc - self.dRestrictiveNodata) < 1e-6):
                    continue


                # 如果限制发展
                if (pc < 1e-6):
                    continue

                pg = self.pppProbability[:, r, c]
                pn = self.calNeighbor(r, c)
                #pr = np.random.uniform()
                pd = pg*(0.5+0.5*pn)*(0.1+pc*0.9)*self.pInertia*self.ppCostMatrix[val-1]

                if(pd.sum() < 1e-6):
                    continue

                ind, p = FLUS_Utils.rouletteWheelSelection(pd)

                if(p < 0.1):
                    continue
                if((self.pDemandAmount[ind] > 0 and self.pCurrentAmount[ind] < self.pTargetAmount[ind]) ):
                    # 更新目前各类像元的数目
                    self.ppSimulateLU[r,c] = ind + 1
                    self.pCurrentAmount[ind] += 1
                    self.pCurrentAmount[val-1] -= 1
                # print (ind, val)
                samplenum += 1

            # 计算目前的发展需求，修正惯性值
            pDemand_t_1 = self.pTargetAmount - self.pCurrentAmount
            for i in range(self.pInertia.shape[0]):
                if(abs(pDemand_t_1[i]) < abs(self.pDemandAmount[i])):
                    continue
                if(self.pDemandAmount[i] < 0 and pDemand_t_1[i] < 0):
                    self.pInertia[i] *= (self.pDemandAmount[i]/pDemand_t_1[i])
                elif(self.pDemandAmount[i] > 0 and pDemand_t_1[i] > 0):
                    self.pInertia[i] *= (pDemand_t_1[i]/self.pDemandAmount[i])

            # 更新发展需求
            self.pDemandAmount = pDemand_t_1
            print ("\t\tinertia: ", self.pInertia)

            nIter += 1

            bFlag = True
            for current, target in zip(self.pCurrentAmount, self.pTargetAmount):
                bFlag = bFlag and abs(current - target) < 1


        self.sendMsg.emit("save file...")
        if (FLUS_Utils.array2Tif(self.ppSimulateLU.reshape(1, self.nRows, self.nCols), self.sSimLUFileName, self.gDataType, self.dLUNodata,
                                     self.pGeoTransform,
                                     self.sProjectionRef) == None):
            self.sendMsg2Box.emit("File Exists!\nPlease Delete it first!", "error")
            self.finished.emit()




    def run(self):
        """
        启动线程
        :return:
        """
        self.parseXML()
        self.init()
        self.iteration()
        self.finished.emit()
        pass